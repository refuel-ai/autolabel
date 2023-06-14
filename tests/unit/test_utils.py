from autolabel import utils
from rich.console import Console

console = Console()


def test_get_data():
    import os

    for dataset in utils.EXAMPLE_DATASETS:
        utils.get_data(dataset)
        os.remove("seed.csv")
        os.remove("test.csv")


def test_maybe_round():
    assert utils.maybe_round(4) == 4
    assert utils.maybe_round(4.123456789) == 4.1235
    assert utils.maybe_round(3.30009) == 3.3001
    assert utils.maybe_round(1000) == 1000
    assert utils.maybe_round(float(0)) == 0.0
    assert utils.maybe_round(0) == 0
    assert utils.maybe_round("test") == "test"


def test_track_with_data():
    indices = range(0, 25, 5)
    for current_index in utils.track_with_stats(
        indices,
        {},
        total=25,
        advance=5,
        console=console,
    ):
        continue


def test_track():
    inputs = list(range(0, 100))
    for input_i in utils.track(
        inputs,
        description="Testing track function...",
        console=console,
    ):
        continue


def test_extract_valid_json_substring():
    test1 = 'hello world {"hello": "world"}'
    assert utils.extract_valid_json_substring(test1) == '{"hello": "world"}'
    test2 = 'hello (*&*{(&^-)*(=)_ {"hello": {"nested": "world"}}'
    assert utils.extract_valid_json_substring(test2) == '{"hello": {"nested": "world"}}'
    test3 = '{"hello": {"double": {"nested": "world"}}} !@#$%^^&**({}{{{"n":_}}}}}}}})'
    assert (
        utils.extract_valid_json_substring(test3)
        == '{"hello": {"double": {"nested": "world"}}}'
    )


def test_calculate_md5():
    import random
    import string

    rand_string = "".join(random.choices(string.printable, k=1000))
    rand_md5 = utils.calculate_md5([rand_string, "more data", {}])
    dict_md5 = utils.calculate_md5({"test": "dictionary"})
    assert utils.calculate_md5("test") == utils.calculate_md5("test")
    file_md5 = utils.calculate_md5(open("tests/assets/banking/test.csv", "rb"))


def test_get_format_variables():
    assert utils.get_format_variables("hello {var1} world {var2}") == ["var1", "var2"]
    assert utils.get_format_variables("(*&*()) {foo} __ {} ... {bar}") == [
        "foo",
        "",
        "bar",
    ]
