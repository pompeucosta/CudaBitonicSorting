/**
 * @brief Recursively merges subsequences in bitonic order using the direction parameter.
 * @param data Pointer to the array containing the data to merge.
 * @param low Starting index of the subsequence within the array.
 * @param cnt Number of elements in the subsequence to merge.
 * @param dir Direction to merge the data (1 for ascending, 0 for descending).
 */
void bitonic_merge(int *data, int low, int cnt, int dir)
{
    if (cnt > 1)
    {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++)
        {
            if ((i + k) < (low + cnt))
            { // Ensuring the index is within bounds
                if (dir == (data[i] > data[i + k]))
                {
                    int temp = data[i];
                    data[i] = data[i + k];
                    data[i + k] = temp;
                }
            }
        }
        bitonic_merge(data, low, k, dir);
        bitonic_merge(data, low + k, k, dir);
    }
}

/**
 * @brief Sorts the data recursively in a bitonic sequence and then merges.
 * @param data Pointer to the array containing the data to sort.
 * @param low Starting index of the subsequence within the array.
 * @param cnt Number of elements in the subsequence to sort.
 * @param dir Direction to sort the data (1 for ascending, 0 for descending).
 */
void bitonic_sort(int *data, int low, int cnt, int dir)
{
    if (cnt > 1)
    {
        int k = cnt / 2;
        bitonic_sort(data, low, k, 1);      // Sort in ascending order
        bitonic_sort(data, low + k, k, 0);  // Sort in descending order
        bitonic_merge(data, low, cnt, dir); // Merge whole sequence into one direction
    }
}