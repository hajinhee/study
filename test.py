from icecream import ic

class Test:
    def __init__(self) -> None:
        pass

    def non_static_method(self):
        ic('non_static_method')

    @staticmethod
    def static_method():
        ic('static_method')



if __name__ == '__main__':
    test_object = Test()
    test_object.non_static_method()
    test_object.static_method()

    test_object2 = Test()
    test_object2.non_static_method()
    test_object2.static_method()

    Test.static_method()
    Test().static_method()
    # Test.non_static_method() 
    Test().non_static_method()