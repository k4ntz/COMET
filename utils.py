from pysr import PySRRegressor

def get_model():
    model = PySRRegressor(
        niterations = 150,  # < Increase me for better results
        maxsize = 15,
        binary_operators = ["+", "-", "max", "min"],
        elementwise_loss = "loss(prediction, target) = (prediction - target)^2",
    )
    return model