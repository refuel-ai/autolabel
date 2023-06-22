"""Test utils"""
from typing import Optional, Any

import tempfile
import os
from autolabel import utils
from rich.console import Console

console = Console()


def test_get_data(mocker) -> None:
    """Test Get Data"""
    dataset_name = "banking"

    def assert_text_remove(file_name_: str, text: str):
        """Assert text and remove temp file

        Args:
            file_name_ (str): Temporary file name
            text (str): text to check
        """
        with open(file_name_, "r") as tmp_file_read:
            file_content = tmp_file_read.read()
            assert file_content == text
        os.remove(file_name)

    def generate_tempfile_with_content(
        input_url: str, bar: Optional[Any] = None
    ) -> None:
        """Generate a Temporary file with dummy content"""
        with tempfile.NamedTemporaryFile(dir="./", delete=False) as tmp_file:
            file_name = os.path.basename(input_url)
            os.rename(tmp_file.name, file_name)
            tmp_file.write(f"{input_url}".encode("utf-8"))
            tmp_file.flush()

    mocker.patch("wget.download", side_effect=generate_tempfile_with_content)

    # The below case handles the case when the force argument is not provided
    # We create two dummy files and insert text to it.
    # Since the files are already present download should not happen
    for file_name in ["seed.csv", "test.csv"]:
        generate_tempfile_with_content(f"temp_download_without_force/{file_name}")

    utils.get_data(dataset_name=dataset_name)

    for file_name in ["seed.csv", "test.csv"]:
        assert_text_remove(file_name, text=f"temp_download_without_force/{file_name}")

    # The below case handles the case when the force argument is provided
    # We create two dummy files and insert text to it.
    # Despite the files are already present, the files are deleted and new files
    # are downloaded. The below case assert that the contents match the new files.
    for file_name in ["seed.csv", "test.csv"]:
        generate_tempfile_with_content(f"temp_download_with_force/{file_name}")

    utils.get_data(dataset_name=dataset_name, force=True)

    for file_name in ["seed.csv", "test.csv"]:
        assert_text_remove(
            file_name,
            text=utils.DATASET_URL.format(
                dataset=dataset_name, partition=file_name[0:-4]
            ),
        )


def test_maybe_round():
    assert utils.maybe_round(4) == 4
    assert utils.maybe_round(4.123456789) == 4.1235
    assert utils.maybe_round(3.30009) == 3.3001
    assert utils.maybe_round(1000) == 1000
    assert utils.maybe_round(float(0)) == 0.0
    assert utils.maybe_round(0) == 0
    assert utils.maybe_round("test") == "test"


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
