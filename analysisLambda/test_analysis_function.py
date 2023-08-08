import pickle

from analysisLambda import get_from_quickfs, to_json

def load_test_json():
    with open("analysis.pkl", "rb") as f:
        return pickle.load(f)

def test_scrape_to_json():
    data, ks = get_from_quickfs(load_test_json)
    j = to_json(data)
    assert "pb" in ks
    assert all([str(y) in j.keys() for y in range(2013, 2023)])
