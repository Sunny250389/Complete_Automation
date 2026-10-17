"""OpenAPI 3.0 / 3.1 contract validation of requests and responses.

The document is checked with ``openapi-spec-validator``; payloads are checked
with ``openapi-schema-validator`` (OpenAPI's schema dialects, including 3.0
``nullable``). ``$ref`` is resolved within the document.

Supported: path, query and header parameters (``simple``/``form`` styles,
primitives and arrays), request bodies, documented status codes (exact,
``2XX`` ranges, ``default``), required response headers, content types and
JSON bodies. Not validated: cookie parameters, ``deepObject``/delimited
styles, non-JSON body schemas, ``discriminator`` mapping beyond JSON Schema,
callbacks, links and security requirements. Remote ``$ref`` (other files or
URLs) is not followed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx
import yaml
from openapi_schema_validator import (
    OAS30Validator,
    OAS31Validator,
    oas30_format_checker,
    oas31_format_checker,
)
from openapi_spec_validator import validate as validate_spec
from openapi_spec_validator.validation.exceptions import OpenAPISpecValidatorError, ValidatorDetectError
from jsonschema.exceptions import ValidationError as SchemaValidationError
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT4, DRAFT202012

from complete_automation.api.client import ApiResponse, path_template_regex
from complete_automation.api.errors import ContractError
from complete_automation.api.schema import describe_error
from complete_automation.core.datasets import load_json, load_yaml
from complete_automation.core.exceptions import TestDataError
from complete_automation.core.redaction import DEFAULT_REDACTOR, Redactor

_URI = "urn:complete-automation:openapi"
_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")
_ABSENT: Any = object()
MAX_REMOTE_BYTES = 5_000_000


@dataclass(frozen=True)
class ContractViolation:
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message}"


@dataclass(frozen=True)
class Operation:
    method: str
    path: str
    """The documented path template, e.g. ``/users/{user_id}``."""
    pointer: str
    """JSON pointer of the operation object in the document."""
    concrete: str = ""
    """The request path that matched, without server base path or query (still percent-encoded)."""


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _mime(content_type: str | None) -> str | None:
    return content_type.split(";")[0].strip().lower() if content_type else None


def _is_json(mime: str | None) -> bool:
    return bool(mime) and (mime == "application/json" or mime.endswith("+json"))  # type: ignore[union-attr]


class OpenApiContract:
    """A validated OpenAPI document with request and response checks."""

    def __init__(self, spec: Mapping[str, Any], *, source: str = "<document>", redactor: Redactor | None = None) -> None:
        if not isinstance(spec, Mapping):
            raise ContractError(f"OpenAPI document {source} must be an object")
        try:
            validate_spec(dict(spec))
        except (OpenAPISpecValidatorError, SchemaValidationError, ValidatorDetectError) as error:
            message = getattr(error, "message", str(error)).splitlines()[0]
            raise ContractError(f"invalid OpenAPI document {source}: {message}") from None
        version = str(spec.get("openapi", ""))
        if version.startswith("3.0"):
            self._validator, self._formats, dialect = OAS30Validator, oas30_format_checker, DRAFT4
        elif version.startswith("3.1"):
            self._validator, self._formats, dialect = OAS31Validator, oas31_format_checker, DRAFT202012
        else:
            raise ContractError(f"{source}: OpenAPI {version or '?'} is not supported (use 3.0 or 3.1)")
        self.spec = dict(spec)
        self.source = source
        self.version = version
        self.redactor = redactor or DEFAULT_REDACTOR
        self._registry = Registry().with_resource(_URI, Resource.from_contents(self.spec, default_specification=dialect))
        self._base_paths = sorted(
            {httpx.URL(s.get("url", "/")).path.rstrip("/") for s in self.spec.get("servers", []) if isinstance(s, dict)} - {""},
            key=len,
            reverse=True,
        )

    # -- loading -----------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> OpenApiContract:
        path = Path(path)
        try:
            spec = load_yaml(path) if path.suffix.lower() in (".yaml", ".yml") else load_json(path)
        except TestDataError as error:
            raise ContractError(f"cannot load OpenAPI document {path}: {error}") from error
        return cls(spec, source=str(path), **kwargs)

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        timeout: float = 10.0,
        verify: bool = True,
        transport: httpx.BaseTransport | None = None,
        **kwargs: Any,
    ) -> OpenApiContract:
        """Fetch a document over HTTP(S). The content is parsed as JSON or YAML, never executed."""
        try:
            with httpx.Client(timeout=timeout, verify=verify, transport=transport, trust_env=False,
                              follow_redirects=True, max_redirects=3) as client:
                response = client.get(url, headers={"Accept": "application/json, application/yaml"})
        except httpx.HTTPError as error:
            raise ContractError(f"cannot fetch OpenAPI document {url}: {type(error).__name__}") from error
        if response.status_code != 200:
            raise ContractError(f"cannot fetch OpenAPI document {url}: HTTP {response.status_code}")
        if len(response.content) > MAX_REMOTE_BYTES:
            raise ContractError(f"OpenAPI document {url} is larger than {MAX_REMOTE_BYTES} bytes")
        try:
            spec = json.loads(response.content)
        except ValueError:
            try:
                spec = yaml.safe_load(response.content)
            except yaml.YAMLError:
                raise ContractError(f"OpenAPI document {url} is neither JSON nor YAML") from None
        return cls(spec, source=url, **kwargs)

    # -- lookup ------------------------------------------------------------------------

    def _get(self, pointer: str) -> Any:
        node: Any = self.spec
        for token in pointer.split("/")[1:]:
            token = token.replace("~1", "/").replace("~0", "~")
            node = node[int(token)] if isinstance(node, list) else node[token]
        return node

    def _deref(self, pointer: str) -> tuple[Any, str]:
        """Follow local ``$ref`` chains; returns the object and its own pointer."""
        node = self._get(pointer)
        seen = set()
        while isinstance(node, dict) and isinstance(node.get("$ref"), str) and node["$ref"].startswith("#"):
            if pointer in seen:
                raise ContractError(f"{self.source}: circular $ref at {pointer}")
            seen.add(pointer)
            pointer = node["$ref"][1:]
            node = self._get(pointer)
        return node, pointer

    def operations(self) -> list[tuple[str, str]]:
        return [
            (method.upper(), path)
            for path, item in self.spec.get("paths", {}).items()
            for method in _METHODS
            if isinstance(item, dict) and method in item
        ]

    def find_operation(self, method: str, path: str) -> Operation | None:
        """The operation for a concrete request path (server base paths are stripped)."""
        method = method.lower()
        path = path.split("?", 1)[0] or "/"
        stripped = [path[len(base):] or "/" for base in self._base_paths if path.startswith(base + "/") or path == base]
        templates = sorted(self.spec.get("paths", {}), key=lambda t: (t.count("{"), -len(t)))
        for concrete in [*stripped, path]:
            for template in templates:
                item = self.spec["paths"][template]
                if method in item and path_template_regex(template).match(concrete):
                    return Operation(method.upper(), template, f"/paths/{_escape(template)}/{method}", concrete)
        return None

    def _parameters(self, operation: Operation) -> list[tuple[dict[str, Any], str]]:
        found: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
        item_pointer = operation.pointer.rsplit("/", 1)[0]
        for owner in (item_pointer, operation.pointer):
            for index, _ in enumerate(self._get(owner).get("parameters", [])):
                param, pointer = self._deref(f"{owner}/parameters/{index}")
                found[(param["name"].lower() if param["in"] == "header" else param["name"], param["in"])] = (param, pointer)
        return list(found.values())

    # -- validation --------------------------------------------------------------------

    def _check(self, instance: Any, schema_pointer: str, location: str) -> list[ContractViolation]:
        validator = self._validator({"$ref": f"{_URI}#{schema_pointer}"}, registry=self._registry, format_checker=self._formats)
        errors = sorted(validator.iter_errors(instance), key=lambda e: [str(p) for p in e.absolute_path])
        return [
            ContractViolation(v.location, v.message)
            for v in (describe_error(error, location, self.redactor) for error in errors)
        ]

    def _coerce(self, value: Any, schema_pointer: str) -> Any:
        schema, pointer = self._deref(schema_pointer)
        kind = schema.get("type")
        if isinstance(kind, list):
            kind = next((k for k in kind if k != "null"), None)
        if kind == "array":
            values = value if isinstance(value, list) else str(value).split(",")
            return [self._coerce(item, f"{pointer}/items") for item in values] if "items" in schema else values
        if isinstance(value, list):
            value = value[-1]
        try:
            if kind == "integer" and str(value).lstrip("-").isdigit():
                return int(value)
            if kind == "number":
                return float(value)
            if kind == "boolean" and str(value).lower() in ("true", "false"):
                return str(value).lower() == "true"
        except ValueError:
            pass
        return value

    def validate_request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str | list[str]] | None = None,
        headers: Mapping[str, str] | None = None,
        body: Any = _ABSENT,
        content_type: str | None = None,
    ) -> list[ContractViolation]:
        operation = self.find_operation(method, path)
        if operation is None:
            return [ContractViolation(f"{method.upper()} {path}", "operation is not documented")]
        violations: list[ContractViolation] = []
        match = path_template_regex(operation.path).match(operation.concrete)
        path_values = {k: unquote(v) for k, v in (match.groupdict() if match else {}).items()}
        query = dict(query or {})
        headers_lower = {k.lower(): v for k, v in (headers or {}).items()}
        sources: dict[str, Mapping[str, Any]] = {"path": path_values, "query": query, "header": headers_lower}
        for param, pointer in self._parameters(operation):
            where, name = param["in"], param["name"]
            source = sources.get(where)
            if source is None:
                continue  # cookie parameters are not validated
            key = name.lower() if where == "header" else name
            location = f"request.{where}.{name}"
            if key not in source:
                if param.get("required", where == "path"):
                    violations.append(ContractViolation(location, "required parameter is missing"))
                continue
            if "schema" in param:
                value = self._coerce(source[key], f"{pointer}/schema")
                violations += self._check(value, f"{pointer}/schema", location)
        violations += self._validate_body(operation, body, content_type)
        return violations

    def _validate_body(self, operation: Operation, body: Any, content_type: str | None) -> list[ContractViolation]:
        operation_object = self._get(operation.pointer)
        if "requestBody" not in operation_object:
            return []
        request_body, pointer = self._deref(f"{operation.pointer}/requestBody")
        if body is _ABSENT:
            return [ContractViolation("request.body", "required body is missing")] if request_body.get("required") else []
        return self._validate_content(request_body.get("content", {}), f"{pointer}/content", body, content_type, "request.body")

    def _validate_content(
        self, content: dict[str, Any], pointer: str, body: Any, content_type: str | None, location: str
    ) -> list[ContractViolation]:
        mime = _mime(content_type)
        key = next(
            (k for k in content if k.lower() == mime),
            next((k for k in content if mime and k.endswith("/*") and mime.startswith(k[:-1].lower())), None),
        ) or ("*/*" if "*/*" in content else None)
        if key is None:
            return [ContractViolation(location, f"content type {mime or 'missing'} is not documented (documented: {', '.join(content) or 'none'})")]
        if _is_json(mime) and "schema" in content[key]:
            return self._check(body, f"{pointer}/{_escape(key)}/schema", location)
        return []

    def validate_response(
        self,
        response: ApiResponse | None = None,
        *,
        method: str | None = None,
        path: str | None = None,
        status_code: int | None = None,
        headers: Mapping[str, str] | None = None,
        body: Any = _ABSENT,
        content_type: str | None = None,
    ) -> list[ContractViolation]:
        """Check a response against the documented responses of its operation."""
        if response is not None:
            method, path, status_code = response.method, response.raw.request.url.raw_path.decode(), response.status_code
            headers, content_type = dict(response.headers), response.headers.get("Content-Type")
            if response.content:
                body = response.json() if response.is_json else response.content
        if not (method and path and status_code is not None):
            raise ValueError("give a response, or method, path and status_code")
        operation = self.find_operation(method, path)
        if operation is None:
            return [ContractViolation(f"{method.upper()} {path}", "operation is not documented")]
        responses = self._get(operation.pointer).get("responses", {})
        code = next(
            (k for k in (str(status_code), f"{status_code // 100}XX", f"{status_code // 100}xx", "default") if k in responses),
            None,
        )
        if code is None:
            return [ContractViolation("response.status", f"{status_code} is not documented for {method.upper()} {operation.path} (documented: {', '.join(responses)})")]
        documented, pointer = self._deref(f"{operation.pointer}/responses/{_escape(code)}")
        violations: list[ContractViolation] = []
        present = {k.lower() for k in (headers or {})}
        for name, _ in documented.get("headers", {}).items():
            header, _pointer = self._deref(f"{pointer}/headers/{_escape(name)}")
            if header.get("required") and name.lower() not in present and name.lower() != "content-type":
                violations.append(ContractViolation(f"response.header.{name}", "required header is missing"))
        content = documented.get("content", {})
        if body is not _ABSENT:
            if not content:
                violations.append(ContractViolation("response.body", f"a body was returned but none is documented for {code}"))
            else:
                violations += self._validate_content(content, f"{pointer}/content", body, content_type, "response.body")
        return violations

    def validate_exchange(self, response: ApiResponse) -> list[ContractViolation]:
        """Validate the request that produced ``response`` and the response itself."""
        request = response.raw.request
        grouped: dict[str, list[str]] = {}
        for key, value in request.url.params.multi_items():
            grouped.setdefault(key, []).append(value)
        query: dict[str, str | list[str]] = {k: v[0] if len(v) == 1 else v for k, v in grouped.items()}
        body: Any = _ABSENT
        try:
            if request.content:
                body = json.loads(request.content) if _is_json(_mime(request.headers.get("Content-Type"))) else request.content
        except (httpx.RequestNotRead, ValueError):
            body = _ABSENT
        return self.validate_request(
            request.method, request.url.raw_path.decode(), query=query, headers=dict(request.headers),
            body=body, content_type=request.headers.get("Content-Type"),
        ) + self.validate_response(response)

    def assert_response(self, response: ApiResponse, *, include_request: bool = False) -> None:
        """Fail with every contract violation of the response (and optionally its request)."""
        violations = self.validate_exchange(response) if include_request else self.validate_response(response)
        if violations:
            shown = "\n".join(f"  - {v}" for v in violations[:20])
            raise AssertionError(f"{response.describe()}: breaks the contract {self.source}:\n{shown}")

    # -- compatibility -----------------------------------------------------------------

    def _top_level(self, schema_pointer: str) -> tuple[set[str], set[str]]:
        schema, _ = self._deref(schema_pointer)
        return set(schema.get("properties", {})), set(schema.get("required", []))


def find_breaking_changes(old: OpenApiContract, new: OpenApiContract) -> list[str]:
    """Changes in ``new`` that can break clients written against ``old``.

    Detected: removed operations, removed documented status codes, new required
    parameters, a request body that became required, new required top-level
    request properties, and top-level properties removed from JSON 2xx responses.
    Nested schema changes are not compared.
    """
    changes: list[str] = []
    new_operations = set(new.operations())
    for method, path in old.operations():
        name = f"{method} {path}"
        if (method, path) not in new_operations:
            changes.append(f"{name}: operation removed")
            continue
        old_op = Operation(method, path, f"/paths/{_escape(path)}/{method.lower()}")
        new_op = old_op
        old_obj, new_obj = old._get(old_op.pointer), new._get(new_op.pointer)
        for code in sorted(set(old_obj.get("responses", {})) - set(new_obj.get("responses", {}))):
            changes.append(f"{name}: response {code} no longer documented")
        old_required = {(p["in"], p["name"]) for p, _ in old._parameters(old_op) if p.get("required")}
        for param, _ in new._parameters(new_op):
            if param.get("required") and (param["in"], param["name"]) not in old_required:
                changes.append(f"{name}: new required {param['in']} parameter '{param['name']}'")
        if "requestBody" in new_obj:
            new_body, new_pointer = new._deref(f"{new_op.pointer}/requestBody")
            old_body = old._deref(f"{old_op.pointer}/requestBody")[0] if "requestBody" in old_obj else {}
            if new_body.get("required") and not old_body.get("required"):
                changes.append(f"{name}: request body became required")
            if "application/json" in new_body.get("content", {}) and "schema" in new_body["content"]["application/json"]:
                _, new_req = new._top_level(f"{new_pointer}/content/application~1json/schema")
                old_req: set[str] = set()
                if "application/json" in old_body.get("content", {}) and "schema" in old_body["content"]["application/json"]:
                    _, old_pointer = old._deref(f"{old_op.pointer}/requestBody")
                    _, old_req = old._top_level(f"{old_pointer}/content/application~1json/schema")
                for prop in sorted(new_req - old_req):
                    changes.append(f"{name}: request property '{prop}' became required")
        for code, _ in old_obj.get("responses", {}).items():
            if not str(code).startswith("2") or code not in new_obj.get("responses", {}):
                continue
            old_resp, old_pointer = old._deref(f"{old_op.pointer}/responses/{_escape(code)}")
            new_resp, new_pointer = new._deref(f"{new_op.pointer}/responses/{_escape(code)}")
            if "schema" in old_resp.get("content", {}).get("application/json", {}) and "schema" in new_resp.get("content", {}).get("application/json", {}):
                old_props, _ = old._top_level(f"{old_pointer}/content/application~1json/schema")
                new_props, _ = new._top_level(f"{new_pointer}/content/application~1json/schema")
                for prop in sorted(old_props - new_props):
                    changes.append(f"{name}: response {code} no longer returns '{prop}'")
    return changes
