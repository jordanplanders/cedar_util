import types




def is_float(string):
    try:
        val = float(string)
    except:
        val= None
    return val


def filter_quantiles(values, lower_quantile=0.01, upper_quantile=0.99):
    """
    Filter values between specified quantiles.

    Args:
    values: Array-like object of values.
    lower_quantile: Lower bound quantile.
    upper_quantile: Upper bound quantile.

    Returns:
    Filtered values.
    """
    quants = np.quantile(values, [lower_quantile, upper_quantile])
    return values[(values > quants[0]) & (values < quants[1])]


class Bootstrapped:
    def __init__(self):
        # No need for a pools dictionary anymore
        pass

    def add(self, sampling_object):
        """
        Add a sampling object to the instance attributes.
        """
        for attr, value in sampling_object.__dict__.items():
            if not hasattr(self, attr):
                setattr(self, attr, [])
            getattr(self, attr).append(value)

    def get_pool(self, attr_name):
        """
        Retrieve the collection of a specific attribute.
        """
        return getattr(self, attr_name, [])

    def get_all_pools(self):
        """
        Retrieve all attributes that have been pooled.
        """
        return {attr: getattr(self, attr) for attr in self.__dict__.keys()}


    def add_method(self, method_name, method_function):
        """
        Dynamically add a method to this instance.
        """
        bound_method = types.MethodType(method_function, self)
        setattr(self, method_name, bound_method)