from tqdm import tqdm

def print_step(iterable, prefix="[Sampler]", total=None):    
    """
    Wraps an iterable with a tqdm progress bar.

    Args:
        iterable (iterable): The loop to wrap.
        prefix (str): A label to prepend in the bar.
        total (int): Optional override of total steps.

    Returns:
        tqdm iterable
    """
    for i in tqdm(iterable, desc=prefix, total=total):
        yield i
    #return tqdm(iterable, desc=prefix, total=total)
