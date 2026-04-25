-- Person 1 / reference §5.3: optional indexes to drop after TPC-H load so Level 1–2
-- tasks expose Seq Scan optimization opportunities. Safe to run repeatedly.

DROP INDEX IF EXISTS idx_orders_status;
DROP INDEX IF EXISTS idx_lineitem_shipdate;
