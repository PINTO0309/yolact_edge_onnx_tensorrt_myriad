import numpy as np
import tensorflow as tf

data = np.asarray(
    [
        [
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
            ],
            [
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
            ],
            [
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
            ]
        ],
        [
            [
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
            ],
            [
                110,
                111,
                112,
                113,
                114,
                115,
                116,
                117,
                118,
                119,
            ],
            [
                120,
                121,
                122,
                123,
                124,
                125,
                126,
                127,
                128,
                129,
            ]
        ]
    ]
)

indices = np.asarray(
    [
        [0,0,5],
        [1,2,6],
    ]
)

print(f'data.shape: {data.shape}')
print(f'indices.shape: {indices.shape}')

ans = tf.gather_nd(
    data, indices, batch_dims=0
)

print(ans)
print(ans.shape)