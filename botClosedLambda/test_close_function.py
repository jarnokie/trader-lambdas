import pytest

from test_utils import get_table_name, create_trades_table, execute_sql, get_database_config

from botClosedLambda import get_closed_trades

table_name = ""

def end_test():
    """Cleanup the test data in the configured postgres instance."""
    global table_name
    execute_sql([(f"DROP TABLE {table_name};", None)])

@pytest.fixture(scope='session', autouse=True)
def setup_test(request):
    """Initialize the test data in the configured postgres instance to the random table generated from get_table_name()."""
    global table_name
    table_name = get_table_name()
    create_trades_table(table_name)
    request.addfinalizer(end_test)

def test_closed_data():
    """Test the get_open_trades function."""
    global table_name
    db_config = get_database_config()
    open_trades = get_closed_trades(db_config["PG_DB"], db_config["PG_HOST"], db_config["PG_USER"], db_config["PG_PASSWORD"], table_name)
    # Test data set contains 8 trades
    # All of the should NOT have closedate or closeprice set
    assert len(open_trades) == 55
    for trade in open_trades:
        assert trade["closedate"] is not None
        assert trade["closeprice"] is not None
