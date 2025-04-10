def chunk_list(lst, num_chunks):
    k, m = divmod(len(lst), num_chunks)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_chunks)]
