import concurrent
import yt_title_generator.utils.progress as progress_utils
import collections
import yt_title_generator.context as context
import traceback
import sys


def stack_stats(s):
    """Formats the given stack trace"""
    res = []
    this_file = __name__.split(".")[-1] + ".py"
    for file_name, line_number, func_name, _ in s:
        if file_name.endswith(this_file):
            continue
        res.append(f"{func_name}@{file_name}:{line_number}")
    return res


def run(fn, inputs, num_workers):
    """Executes function fn for each input in num_workers threads"""
    exceptions = collections.defaultdict(int)

    def try_fn(x):
        try:
            fn(x)
        except Exception as ex:
            exceptions[
                type(ex).__name__
                + " "
                + str(stack_stats(traceback.extract_tb(sys.exc_info()[2], limit=20)))
            ] += 1

    if num_workers == 1:
        for i in inputs:
            try_fn(i)
        return exceptions

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    executor.map(try_fn, inputs)
    executor.shutdown(wait=True)

    return exceptions


def run_with_progress(fn, inputs, num_workers, progress_text):
    """Ditto but with a progress bar"""

    bar = progress_utils.LockedBar(progress_text, len(inputs))

    def fn_with_progress(x):
        try:
            fn(x)
        except:
            raise
        finally:
            bar.next()

    exceptions = run(fn_with_progress, inputs, num_workers)

    bar.finish()

    return exceptions


def run_main(main_fn, parser):
    """Adds common parameters to command line parser and executes main fn"""
    parser.add_argument(
        "--context", type=str, help="path to context file", required=True
    )
    parser.add_argument("--run", type=str, help="run id", required=True)
    args = parser.parse_args()

    ctx = context.load_context(args.context, args.run)

    exceptions = main_fn(ctx, args)

    if exceptions:
        print("Exceptions:")
        for e, cnt in exceptions.items():
            print(f"\t{e} = {cnt}")
