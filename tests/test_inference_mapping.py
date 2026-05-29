from inference import classes_to_expression


def test_digits_map_to_themselves():
    assert classes_to_expression([4, 11, 3]) == "4+3"


def test_operator_classes():
    # 10 -> '-', 11 -> '+', 12 -> '*'
    assert classes_to_expression([10, 11, 12]) == "-+*"


def test_full_expression():
    assert classes_to_expression([7, 12, 2]) == "7*2"
