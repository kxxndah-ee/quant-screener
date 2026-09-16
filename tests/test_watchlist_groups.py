import unittest
import sqlite3
from src.database.models import (
    init_db,
    get_db_connection,
    update_watchlist_group,
    batch_update_watchlist_groups,
    get_watchlist_groups
)
from src.collectors.krx_universe import batch_add_to_watchlist


class TestWatchlistGroups(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def test_get_groups(self):
        groups = get_watchlist_groups()
        self.assertIsInstance(groups, list)
        self.assertIn('기본그룹', groups)

    def test_update_single_group(self):
        # Update first stock in watchlist
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT code FROM watchlist LIMIT 1')
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            code = row['code']

        ok = update_watchlist_group(code, '테스트그룹')
        self.assertTrue(ok)

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT group_name FROM watchlist WHERE code = ?', (code,))
            self.assertEqual(cursor.fetchone()['group_name'], '테스트그룹')

        # Restore
        update_watchlist_group(code, '기본그룹')

    def test_batch_update_groups(self):
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT code FROM watchlist LIMIT 2')
            codes = [r['code'] for r in cursor.fetchall()]

        cnt = batch_update_watchlist_groups(codes, '배치그룹')
        self.assertEqual(cnt, len(codes))

        # Restore
        batch_update_watchlist_groups(codes, '기본그룹')


if __name__ == '__main__':
    unittest.main()
