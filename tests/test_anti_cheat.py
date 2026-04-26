from sqlsage.anti_cheat import get_result_hash, validate_read_only_sql


def test_validate_read_only_sql_allows_select_and_cte():
    assert validate_read_only_sql("SELECT 1")
    assert validate_read_only_sql("WITH x AS (SELECT 1) SELECT * FROM x")


def test_validate_read_only_sql_blocks_mutations_and_ddl():
    assert not validate_read_only_sql("UPDATE users SET name = 'x'")
    assert not validate_read_only_sql("CREATE INDEX idx ON t (c)")


def test_result_hash_is_deterministic():
    rows = [(1, "a"), (2, "b")]
    assert get_result_hash(rows) == get_result_hash(rows)
