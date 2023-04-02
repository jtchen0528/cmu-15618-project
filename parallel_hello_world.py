import time

def test_pi_loop():
    num_steps = 1600000
    step = 1.0 / num_steps

    the_sum = 0.0

    for j in range(num_steps):
        c = step
        x = ((j-1) - 0.5) * step
        the_sum += 4.0 / (1.0 + x * x)

    pi = step * the_sum
    return pi

import multiprocessing
def calculate_sum(start, end, step):
    local_sum = 0.0
    for j in range(start, end):
        x = ((j-1) - 0.5) * step
        local_sum += 4.0 / (1.0 + x * x)
    return local_sum

def test_pi_loop_mp():
    num_steps = 1600000
    step = 1.0 / num_steps

    the_sum = 0.0

    num_processes = 16
    pool = multiprocessing.Pool(processes=num_processes)

    chunk_size = num_steps // num_processes
    results = []
    for i in range(num_processes):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if i == num_processes - 1:
            end = num_steps
        results.append(pool.apply_async(calculate_sum, args=(start, end, step)))

    pool.close()
    pool.join()

    the_sum = sum(result.get() for result in results)
    pi = step * the_sum
    return pi

import threading
def test_pi_loop_thread():
    num_steps = 1600000
    step = 1.0 / num_steps

    the_sum = 0.0
    lock = threading.Lock()

    def calculate_sum(start, end):
        local_sum = 0.0
        for j in range(start, end):
            x = ((j-1) - 0.5) * step
            local_sum += 4.0 / (1.0 + x * x)
        with lock:
            nonlocal the_sum
            the_sum += local_sum

    num_threads = 16
    threads = []
    chunk_size = num_steps // num_threads
    for i in range(num_threads):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if i == num_threads - 1:
            end = num_steps
        thread = threading.Thread(target=calculate_sum, args=(start, end))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    pi = step * the_sum
    return pi

import concurrent.futures
def test_pi_loop_concurrent():
    num_steps = 1600000
    step = 1.0 / num_steps

    the_sum = 0.0

    def calculate_sum(start, end):
        local_sum = 0.0
        for j in range(start, end):
            x = ((j-1) - 0.5) * step
            local_sum += 4.0 / (1.0 + x * x)
        return local_sum

    num_workers = 16
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        chunk_size = num_steps // num_workers
        futures = []
        for i in range(num_workers):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            if i == num_workers - 1:
                end = num_steps
            future = executor.submit(calculate_sum, start, end)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            the_sum += future.result()

    pi = step * the_sum
    return pi


start_time = time.time()
res = test_pi_loop()
print(f'"--- Serial --- \n res = {res} \n runtime: {time.time() - start_time} seconds \n"')
start_time = time.time()
res = test_pi_loop_mp()
print(f'"--- Multiprocessing --- \n res = {res} \n runtime: {time.time() - start_time} seconds \n"')
start_time = time.time()
res = test_pi_loop_thread()
print(f'"--- Thread --- \n res = {res} \n runtime: {time.time() - start_time} seconds \n"')
start_time = time.time()
res = test_pi_loop_concurrent()
print(f'"--- Concurrent --- \n res = {res} \n runtime: {time.time() - start_time} seconds \n"')