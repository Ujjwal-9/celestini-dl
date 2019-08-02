def sub_lists(list1):

    # store all the sublists
    sublist = []

    # first loop
    for i in range(len(list1) + 1):

        # second loop
        for j in range(i + 1, len(list1) + 1):

            # slice the subarray
            sub = list1[i:j]
            sublist.append(sub)


    return sublist
def GoodSubArray (arr, is_bad):
    # Write your code here
    ar = list(arr)
    array = []
    extras = []
    for i in range(len(ar)):
        if i == 0:
            array.append(ar[i])
        else:
            if ar[i] > array[-1]:
                array.append(ar[i])
            else:
                extras.append([ar[i]])
    subs = sub_lists(array)
    total_sub_arrays = len(subs) + len(extras)
    bad_sub_arrays = []
    bad = list(is_bad)
    bads = 0
    for index,i in enumerate(bad):
        if i == 1:
            bad_array = []
            for j in range(index,len(ar)):
                if j == index:
                    bad_array.append(ar[j])
                else:
                    if ar[j] > bad_array[-1]:
                        bad_array.append(ar[j])
            bads += len(bad_array)
    if total_sub_arrays - bads < 0:
        return 0
    else:
        return total_sub_arrays - bads


T = int(input())
for _ in range(T):
    N = int(input())
    arr = map(int, input().split())
    is_bad = map(int, input().split())

    out_ = GoodSubArray(arr, is_bad)
    print (out_)
