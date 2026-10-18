"""A small example schema, used by the documentation and the test suites.

Three related tables show the features the validators cover:

- ``customers``: surrogate key, unique ``email``, NOT NULL columns, server
  defaults, a CHECK on ``status``.
- ``orders``: foreign key to ``customers`` with ``ON DELETE RESTRICT``, an
  index on ``customer_id`` and a stored ``total`` (a derived field: the sum of
  its items' ``amount``).
- ``order_items``: foreign key to ``orders`` with ``ON DELETE CASCADE``, a
  CHECK on ``quantity``, a stored line ``amount`` and a composite unique constraint.

Nothing in the framework depends on this schema; it is a fixture, defined with
SQLAlchemy Core so it creates on SQLite and PostgreSQL alike.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    UniqueConstraint,
    func,
    insert,
)

from complete_automation.core.datasets import DataFactory
from complete_automation.database.validations.schema import (
    ColumnExpectation,
    ForeignKeyExpectation,
    IndexExpectation,
    SchemaExpectation,
    TableExpectation,
)

if TYPE_CHECKING:
    from complete_automation.database.session import DatabaseSession

metadata = MetaData()

customers = Table(
    "customers",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("email", String(255), nullable=False),
    Column("name", String(100), nullable=False),
    Column("status", String(20), nullable=False, server_default="active"),
    Column("created_at", DateTime, nullable=False, server_default=func.current_timestamp()),
    UniqueConstraint("email", name="uq_customers_email"),
    CheckConstraint("status in ('active', 'inactive')", name="ck_customers_status"),
)

orders = Table(
    "orders",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("customer_id", Integer, ForeignKey("customers.id", ondelete="RESTRICT"), nullable=False),
    Column("total", Numeric(10, 2), nullable=False),
    Column("status", String(20), nullable=False, server_default="new"),
    Column("placed_at", DateTime, nullable=False),
    Index("ix_orders_customer_id", "customer_id"),
)

order_items = Table(
    "order_items",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("order_id", Integer, ForeignKey("orders.id", ondelete="CASCADE"), nullable=False),
    Column("sku", String(40), nullable=False),
    Column("quantity", Integer, nullable=False),
    Column("amount", Numeric(10, 2), nullable=False),
    UniqueConstraint("order_id", "sku", name="uq_order_items_order_sku"),
    CheckConstraint("quantity > 0", name="ck_order_items_quantity"),
)

EXAMPLE_TABLES = ("customers", "orders", "order_items")

EXAMPLE_EXPECTATION = SchemaExpectation(
    tables=(
        TableExpectation(
            name="customers",
            columns=(
                ColumnExpectation(name="id", type="integer", primary_key=True, nullable=False),
                ColumnExpectation(name="email", type="string", max_length=255, nullable=False),
                ColumnExpectation(name="name", type="string", max_length=100, nullable=False),
                ColumnExpectation(name="status", type="string", nullable=False, default="active"),
                ColumnExpectation(name="created_at", type="datetime", nullable=False, has_default=True),
            ),
            primary_key=("id",),
            unique_constraints=(("email",),),
        ),
        TableExpectation(
            name="orders",
            columns=(
                ColumnExpectation(name="id", type="integer", primary_key=True),
                ColumnExpectation(name="customer_id", type="integer", nullable=False),
                ColumnExpectation(name="total", type="numeric", nullable=False),
                ColumnExpectation(name="placed_at", type="datetime", nullable=False),
            ),
            primary_key=("id",),
            foreign_keys=(
                ForeignKeyExpectation(
                    columns=("customer_id",), referred_table="customers", referred_columns=("id",), on_delete="RESTRICT"
                ),
            ),
            indexes=(IndexExpectation(columns=("customer_id",), name="ix_orders_customer_id", unique=False),),
        ),
        TableExpectation(
            name="order_items",
            columns=(
                ColumnExpectation(name="id", type="integer", primary_key=True),
                ColumnExpectation(name="order_id", type="integer", nullable=False),
                ColumnExpectation(name="sku", type="string", nullable=False),
                ColumnExpectation(name="quantity", type="integer", nullable=False),
                ColumnExpectation(name="amount", type="numeric", nullable=False),
            ),
            primary_key=("id",),
            foreign_keys=(
                ForeignKeyExpectation(
                    columns=("order_id",), referred_table="orders", referred_columns=("id",), on_delete="CASCADE"
                ),
            ),
            unique_constraints=(("order_id", "sku"),),
        ),
    ),
)

_SKUS = ("SKU-A", "SKU-B", "SKU-C", "SKU-D")
_EPOCH = dt.datetime(2024, 1, 1, 9, 0, 0)


def create_example_schema(db: DatabaseSession) -> None:
    """Create the example tables (needs ``allow_ddl``). Commit afterwards if other connections must see them."""
    db.create_all(metadata)


def drop_example_schema(db: DatabaseSession) -> None:
    """Drop the example tables (needs ``allow_destructive``)."""
    db.drop_all(metadata)


def seed_example_data(
    db: DatabaseSession, *, customers_count: int = 3, orders_per_customer: int = 2, seed: int = 20240101
) -> dict[str, Any]:
    """Insert deterministic rows: the same arguments always produce the same data.

    Order totals equal the sum of their item amounts, so the derived-field checks pass
    on the seed. Returns the number of rows inserted per table.
    """
    factory = DataFactory(seed)
    customer_rows = [
        {"id": n, "email": f"customer{n}.{factory.string(6)}@example.test", "name": f"Customer {n}", "status": "active"}
        for n in range(1, customers_count + 1)
    ]
    order_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    order_id = item_id = 0
    for customer in customer_rows:
        for _ in range(orders_per_customer):
            order_id += 1
            total = Decimal("0.00")
            for sku in _SKUS[: 1 + order_id % len(_SKUS)]:
                item_id += 1
                quantity = factory.integer(1, 4)
                amount = Decimal(factory.integer(100, 5000)) / 100 * quantity
                total += amount
                item_rows.append(
                    {"id": item_id, "order_id": order_id, "sku": sku, "quantity": quantity, "amount": amount}
                )
            order_rows.append(
                {
                    "id": order_id,
                    "customer_id": customer["id"],
                    "total": total,
                    "status": "new",
                    "placed_at": _EPOCH + dt.timedelta(days=order_id),
                }
            )
    if customer_rows:
        db.execute_many(insert(customers), customer_rows)
    if order_rows:
        db.execute_many(insert(orders), order_rows)
    if item_rows:
        db.execute_many(insert(order_items), item_rows)
    if db.manager.dialect == "postgresql":
        # Explicit ids do not advance PostgreSQL sequences; without this the next generated id collides.
        for table_name in EXAMPLE_TABLES:
            db.scalar(
                f"SELECT setval(pg_get_serial_sequence('{table_name}', 'id'), "
                f"(SELECT coalesce(max(id), 1) FROM {table_name}))"
            )  # table names come from the constant above, never from input
    return {"customers": len(customer_rows), "orders": len(order_rows), "order_items": len(item_rows)}
