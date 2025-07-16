import simple_functions


def test_print_ten(capsys):
    simple_functions.print_ten()
    captured = capsys.readouterr()
    assert captured.out.strip() == "1 2 3 4 5 6 7 8 9 10"


def test_print_eleven(capsys):
    simple_functions.print_eleven()
    captured = capsys.readouterr()
    assert captured.out.strip() == "89"


def test_print_twelve(capsys):
    simple_functions.print_twelve()
    captured = capsys.readouterr()
    assert captured.out.strip() == "479001600"
