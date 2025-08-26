import sqlite3
import db_router


def make_cursor():
    conn = sqlite3.connect(":memory:")
    cursor = db_router.LoggedCursor(conn)
    cursor.menace_id = "t"
    return cursor


def test_select_with_join():
    cur = make_cursor()
    sql = 'SELECT u.id FROM "users" u JOIN orders o ON u.id = o.user_id'
    assert cur._table_from_sql(sql) == "users"


def test_select_with_subselect():
    cur = make_cursor()
    sql = 'SELECT * FROM (SELECT * FROM "orders") sub WHERE sub.id = 1'
    assert cur._table_from_sql(sql) == "orders"


def test_insert_with_quotes():
    cur = make_cursor()
    sql = 'INSERT INTO "users" (id) VALUES (1)'
    assert cur._table_from_sql(sql) == "users"


def test_update_with_alias():
    cur = make_cursor()
    sql = 'UPDATE "users" AS u SET name = "a" WHERE u.id = 1'
    assert cur._table_from_sql(sql) == "users"


def test_delete_with_alias():
    cur = make_cursor()
    sql = 'DELETE FROM "users" u WHERE u.id = 1'
    assert cur._table_from_sql(sql) == "users"
