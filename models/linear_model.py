class LinearModel:

    def __init__(self, p0=None, p1=None):

        if p0 is None or p1 is None:
            slope = None
            bias = None
            x = None
            is_vertical = None
        else:
            x0, y0 = p0
            x1, y1 = p1
            is_vertical = x0 == x1

            if is_vertical:
                slope = None
                bias = None
                x = x0
            else:
                slope = (y1 - y0) / (x1 - x0)
                bias = y1 - (x1 * slope)
                x = None

        self.slope = slope
        self.bias = bias
        self.x = x
        self.is_vertical = is_vertical

    def __call__(self, x):

        if self.is_vertical:
            return None

        return (self.slope * x) + self.bias

    def parallel(self, linear_model):

        both_vertical = self.is_vertical and linear_model.is_vertical
        same_slope = self.slope == linear_model.slope

        return both_vertical or same_slope

    def intercept(self, linear_model):

        if self.parallel(linear_model):
            x, y = None, None
        elif self.is_vertical:
            x = self.x
            y = linear_model(x)
        elif linear_model.is_vertical:
            x = linear_model.x
            y = self(x)
        else:
            x = (linear_model.bias - self.bias) / (self.slope - linear_model.slope)
            y = self(x)

        return x, y

    def intersection_point(self, p):

        x, y = p

        if self.slope == 0:
            return x, self.bias
        else:
            perpendicular_slope = -1 / self.slope
            perpendicular_bias = (x / self.slope) + y

            perpendicular = LinearModel()
            perpendicular.slope = perpendicular_slope
            perpendicular.bias = perpendicular_bias
            perpendicular.x = None
            perpendicular.is_vertical = False

            return perpendicular.intercept(self)
