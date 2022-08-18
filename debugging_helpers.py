import inspect
import git


def construct_metadata(brief: bool = False):
    caller_frame_record = inspect.stack()[1]  # 0 represents this line, while 1 represents line at caller
    frame = caller_frame_record[0]
    info = inspect.getframeinfo(frame)

    metadata = dict()

    metadata['filename'] = info.filename  # __FILE__
    metadata['function'] = info.function  # __FUNCTION__
    metadata['lineno'] = info.lineno  # __LINE__

    repo = git.Repo(search_parent_directories=True)
    metadata['commit id'] = repo.head.object.hexsha
    try:
        metadata['branch'] = repo.active_branch.name
    except ValueError:  # TODO: is this the correct thing to catch?
        pass

    if not brief and repo.is_dirty():
        metadata['code diff'] = repo.head.commit.diff(None)

    return metadata

