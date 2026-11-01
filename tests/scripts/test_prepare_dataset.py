"""
Dataset preparation tests

They build a tiny fake dataset, because the real one is not in the
repository.
"""

import pytest

from scripts.prepare_dataset import (find_files, main, position_key, read_dataset,
                                     remove_duplicates, shuffle_boards)

CHECK = "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59"
CHECK_LATER = "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 9 60"
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MATE = "2RRQQ1K/6N1/8/3k4/8/8/8/8 b - - 0 79"


@pytest.fixture
def dataset(tmp_path):
    """A dataset folder sorted by answer, like the real one, with one duplicate."""
    (tmp_path / "check").mkdir()
    (tmp_path / "nothing").mkdir()
    (tmp_path / "checkmate").mkdir()
    (tmp_path / "check" / "10_pieces.txt").write_text(f"{CHECK} Check White\n{CHECK_LATER} Check White\n")
    (tmp_path / "nothing" / "many_pieces.txt").write_text(f"{START} Nothing\n\n")
    (tmp_path / "checkmate" / "10_pieces.txt").write_text(f"{MATE} Checkmate White\n")
    return tmp_path


def test_find_files_gives_the_files_of_the_sub_folders_sorted(dataset):
    (dataset / "clean.txt").write_text("")
    names = [path.split("/")[-2] for path in find_files(str(dataset))]
    assert names == ["check", "checkmate", "nothing"]


def test_find_files_fails_on_a_folder_without_files(tmp_path):
    with pytest.raises(ValueError, match="no chessboard file"):
        find_files(str(tmp_path))


def test_read_dataset_reads_every_board_with_its_answer(dataset):
    boards = read_dataset(find_files(str(dataset)))
    assert boards == [(CHECK, "Check White"), (CHECK_LATER, "Check White"),
                      (MATE, "Checkmate White"), (START, "Nothing")]


def test_read_dataset_stops_on_a_broken_line(dataset):
    (dataset / "check" / "20_pieces.txt").write_text(f"{CHECK} Check Purple\n")
    with pytest.raises(ValueError, match="unknown answer"):
        read_dataset(find_files(str(dataset)))


def test_position_key_ignores_what_the_network_does_not_see():
    assert position_key(CHECK) == position_key(CHECK_LATER)
    assert position_key(CHECK) != position_key(CHECK.replace(" b ", " w "))


def test_remove_duplicates_keeps_the_first_board_of_each_position():
    boards = [(CHECK, "Check White"), (START, "Nothing"), (CHECK_LATER, "Check White")]
    assert remove_duplicates(boards) == [(CHECK, "Check White"), (START, "Nothing")]


def test_shuffle_is_the_same_for_one_seed_and_keeps_every_board():
    boards = [(str(number), "Nothing") for number in range(50)]
    first = shuffle_boards(boards, seed=1)
    assert first == shuffle_boards(boards, seed=1)
    assert first != boards
    assert first != shuffle_boards(boards, seed=2)
    assert sorted(first) == sorted(boards)


def test_main_writes_a_clean_file_ready_for_training(dataset, tmp_path, capsys):
    output = tmp_path / "clean.txt"
    assert main(["--dataset", str(dataset), "--output", str(output), "--seed", "3"]) == 0
    lines = output.read_text().splitlines()
    assert sorted(lines) == sorted([f"{CHECK} Check White", f"{START} Nothing", f"{MATE} Checkmate White"])
    assert "Removed 1 duplicates" in capsys.readouterr().out


def test_main_gives_the_same_file_every_time(dataset, tmp_path):
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    main(["--dataset", str(dataset), "--output", str(first)])
    main(["--dataset", str(dataset), "--output", str(second)])
    assert first.read_text() == second.read_text()


def test_main_returns_84_on_a_missing_folder(tmp_path, capsys):
    assert main(["--dataset", str(tmp_path / "missing"), "--output", str(tmp_path / "clean.txt")]) == 84
    assert "no chessboard file" in capsys.readouterr().err
