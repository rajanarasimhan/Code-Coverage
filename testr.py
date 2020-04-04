from predict import predictx
def test_coverage():
    assert predictx([0,0]) == 0
    assert predictx([0,1]) == 1 
    assert predictx([1,0]) == 1
    assert predictx([1,1]) == 0