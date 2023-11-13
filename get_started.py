from modal import Stub

stub = Stub()


@stub.function()
def square(x: int):
    square: int = x**2
    print(f"This square is: {square}")
