import pickle

try:
    from analysisLambda import get_from_quickfs
except ImportError:
    from lambda_function import get_from_quickfs


def load_test_json():
    with open("analysis.pkl", "rb") as f:
        return pickle.load(f)


def test_scrape_to_json():
    data, ks = get_from_quickfs(load_test_json)
    assert "pb" in ks
    for values in data.values():
        assert all(year in values.keys() for year in range(2013, 2023))


if __name__ == "__main__":
    test_scrape_to_json()
