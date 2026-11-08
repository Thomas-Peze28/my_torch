"""
Dataset preparation tests

They build a tiny fake dataset, because the real one is not in the
repository.
"""

import pytest

from scripts.prepare_dataset import (add_copies, find_files, main, mirror_board, mirror_fen, position_key,
                                     read_dataset, remove_duplicates, shuffle_boards, split_boards,
                                     swap_colors_board, swap_colors_fen)

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
    (dataset / "train.txt").write_text("")
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


def test_mirror_fen_swaps_left_and_right():
    assert mirror_fen("r3k2r/8/8/8/4P3/8/8/R3K2R b KQkq e3 0 1") == "r2k3r/8/8/8/3P4/8/8/R2K3R b - d3 0 1"
    assert mirror_fen(CHECK) == "8/8/4k2R/1p1r4/8/2K5/2P5/8 b - - 7 59"


def test_mirror_fen_twice_gives_the_same_pieces():
    for fen in (CHECK, START, MATE):
        assert position_key(mirror_fen(mirror_fen(fen))) == position_key(fen)


def test_add_copies_keeps_the_answer_and_skips_known_positions():
    symmetric = "8/8/8/3QQ3/8/8/8/8 w - - 0 1"
    train = [(CHECK, "Check White"), (symmetric, "Nothing")]
    others = [(mirror_fen(MATE), "Checkmate White")]
    result = add_copies(train + [(MATE, "Checkmate White")], others, mirror_board)
    assert result == train + [(MATE, "Checkmate White"), (mirror_fen(CHECK), "Check White")]


def test_swap_colors_fen_turns_the_board_and_the_colours():
    assert swap_colors_fen(CHECK) == "8/5p2/5k2/8/4R1P1/r2K4/8/8 w - - 7 59"
    assert swap_colors_fen("r3k2r/8/8/8/4P3/8/8/R3K1R1 b Qk e3 0 1") == "r3k1r1/8/8/4p3/8/8/8/R3K2R w Kq e6 0 1"
    for fen in (CHECK, START, MATE):
        assert swap_colors_fen(swap_colors_fen(fen)) == fen


@pytest.mark.parametrize("answer, swapped", [("Nothing", "Nothing"), ("Check White", "Check Black"),
                                             ("Checkmate Black", "Checkmate White")])
def test_swap_colors_board_swaps_the_colour_of_the_answer(answer, swapped):
    assert swap_colors_board((CHECK, answer)) == (swap_colors_fen(CHECK), swapped)


def test_shuffle_is_the_same_for_one_seed_and_keeps_every_board():
    boards = [(str(number), "Nothing") for number in range(50)]
    first = shuffle_boards(boards, seed=1)
    assert first == shuffle_boards(boards, seed=1)
    assert first != boards
    assert first != shuffle_boards(boards, seed=2)
    assert sorted(first) == sorted(boards)


def test_split_gives_80_10_10_without_losing_or_repeating_a_board():
    boards = [(str(number), "Nothing") for number in range(1000)]
    parts = split_boards(boards)
    assert [len(part) for part in parts.values()] == [800, 100, 100]
    assert parts["train"] + parts["validation"] + parts["test"] == boards


def test_split_shares_can_be_changed():
    parts = split_boards([(str(number), "Nothing") for number in range(10)], validation_share=0.2, test_share=0.3)
    assert [len(part) for part in parts.values()] == [5, 2, 3]


def read_lines(folder, name):
    return (folder / f"{name}.txt").read_text().splitlines()


def test_main_writes_three_files_ready_for_training(dataset, tmp_path, capsys):
    output = tmp_path / "out"
    output.mkdir()
    assert main(["--dataset", str(dataset), "--output", str(output), "--seed", "3"]) == 0
    files = {name: read_lines(output, name) for name in ("train", "validation", "test")}
    every_line = files["train"] + files["validation"] + files["test"]
    assert sorted(every_line) == sorted([f"{CHECK} Check White", f"{START} Nothing", f"{MATE} Checkmate White"])
    assert "Removed 1 duplicates" in capsys.readouterr().out


def lone_kings(number):
    """One of 16 different positions: the white king on the last rank, Black king in a corner."""
    file, side = number % 8, "wb"[number // 8]
    rank = (str(file) if file else "") + "K" + (str(7 - file) if file < 7 else "")
    return f"k7/8/8/8/8/8/8/{rank} {side} - - 0 1"


def test_no_chessboard_is_in_two_files(dataset, tmp_path):
    lines = [f"{lone_kings(number)} Nothing\n" for number in range(16)]
    copies = [line.replace(" 0 1 ", " 4 12 ") for line in lines]
    (dataset / "nothing" / "kings.txt").write_text("".join(lines + copies))

    main(["--dataset", str(dataset), "--output", str(tmp_path)])

    keys = [{position_key(line) for line in read_lines(tmp_path, name)} for name in ("train", "validation", "test")]
    assert sum(len(part) for part in keys) == 16 + 3
    assert not keys[0] & keys[1] and not keys[0] & keys[2] and not keys[1] & keys[2]


def test_main_gives_the_same_files_every_time(dataset, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    main(["--dataset", str(dataset), "--output", str(first)])
    main(["--dataset", str(dataset), "--output", str(second)])
    for name in ("train", "validation", "test"):
        assert read_lines(first, name) == read_lines(second, name)


def test_mirror_only_changes_the_training_file(dataset, tmp_path):
    plain, mirrored = tmp_path / "plain", tmp_path / "mirrored"
    plain.mkdir()
    mirrored.mkdir()
    main(["--dataset", str(dataset), "--output", str(plain)])
    main(["--dataset", str(dataset), "--output", str(mirrored), "--mirror"])

    for name in ("validation", "test"):
        assert read_lines(plain, name) == read_lines(mirrored, name)
    train = read_lines(mirrored, "train")
    assert set(read_lines(plain, "train")) < set(train)
    assert len(train) == 2 * len(read_lines(plain, "train"))
    others = {position_key(line) for name in ("validation", "test") for line in read_lines(mirrored, name)}
    assert not {position_key(line) for line in train} & others


def test_mirror_and_swap_colors_give_four_times_the_training_file(dataset, tmp_path):
    plain, both = tmp_path / "plain", tmp_path / "both"
    plain.mkdir()
    both.mkdir()
    main(["--dataset", str(dataset), "--output", str(plain)])
    main(["--dataset", str(dataset), "--output", str(both), "--mirror", "--swap-colors"])

    assert read_lines(plain, "validation") == read_lines(both, "validation")
    assert len(read_lines(both, "train")) == 4 * len(read_lines(plain, "train"))
    answers = [line.split(" ", 6)[6] for line in read_lines(both, "train")]
    assert answers.count("Check White") == answers.count("Check Black")
    assert answers.count("Checkmate White") == answers.count("Checkmate Black")


def test_main_returns_84_on_a_missing_folder(tmp_path, capsys):
    assert main(["--dataset", str(tmp_path / "missing"), "--output", str(tmp_path)]) == 84
    assert "no chessboard file" in capsys.readouterr().err
