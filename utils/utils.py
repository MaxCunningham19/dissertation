def extract_kwargs(kwargs: list[str]):
    kwargs_dict = {}
    if kwargs:
        for pair in kwargs:
            key, value = pair.split("=")
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except:
                pass
            kwargs_dict[key] = value
    return kwargs_dict
