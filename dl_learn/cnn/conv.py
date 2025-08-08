import numpy as np

def convolution(input, filter, bias, stride=1, pad=0):
    input = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    input_n, channel, height, width = input.shape
    o_channel, f_channel, f_height, f_width = filter.shape

    assert channel == f_channel, "input and filter different channel count"
    assert o_channel == bias.shape[0], "filter and bias has different 0-axis length"

    width_step = (width - f_width) // stride
    height_step = (height - f_height) // stride

    o_height = width_step + 1
    o_width = height_step + 1

    result = []

    for i_n in range(input_n):
        i_result = []
        for o in range(o_channel):
            output = np.zeros((o_height, o_width))

            for c in range(channel):
                for i in range(o_height):
                    for j in range(o_width):
                        origin_x = i * stride
                        origin_y = j * stride

                        sum_res = 0
                        for idx, c in np.ndenumerate(filter[o, c]):
                            new_idx_x = idx[0] + origin_x
                            new_idx_y = idx[1] + origin_y
                            sum_res += input[i_n, c, new_idx_x, new_idx_y] * c

                        output[i, j] += sum_res

            i_result.append(output)

        i_item = np.stack(i_result) + bias
        result.append(i_item)

    return np.stack(result)




# left: 4x4 matrix
# right: 3x3 matrix
def im2col(input, f_n, f_channel, f_height, ):
    pass

if __name__ == "__main__":
    # a = np.random.rand(4, 4)
    # b = np.random.rand(3, 3)

    a = np.array([[1, 2, 3, 0], [0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1]])

    a = np.stack([a, a, a])

    a = np.stack([a, a])
    b = np.array([[2, 0, 1], [0, 1, 2], [1, 0, 2]])

    b = np.stack([b, b, b])
    b = np.stack([b, b])

    bias = np.array([[[1]], [[2]]])

    print(f"shape info: a= {a.shape} b= {b.shape}")
    res = convolution(a, b, bias, pad=1)
    print(f"conv res: {res}")
