from typing import Callable

import torch

# The coefficients of spherical harmonics is from:
# http://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
# Here we generate the coefficients by `C++` `long double` type.
# `tuple` for immutable.
C = (
    # 0
    (
        0.28209479177387814346,  # 1 / (2 * sqrt(pi))
    ),
    # 1
    (
        0.48860251190291992160,  # sqrt(3) / (2 * sqrt(pi))
        0.48860251190291992160,  # sqrt(3) / (2 * sqrt(pi))
        0.48860251190291992160,  # sqrt(3) / (2 * sqrt(pi))
    ),
    # 2
    (
        1.09254843059207907056,  # sqrt(15) / (2 * sqrt(pi))
        1.09254843059207907056,  # sqrt(15) / (2 * sqrt(pi))
        0.31539156525252000604,  # sqrt(5) / (4 * sqrt(pi))
        1.09254843059207907056,  # sqrt(15) / (2 * sqrt(pi))
        0.54627421529603953528,  # sqrt(15) / (4 * sqrt(pi))
    ),
    # 3
    (
        0.59004358992664351035,  # sqrt(70) / (8 * sqrt(pi))
        2.89061144264055405544,  # sqrt(105) / (2 * sqrt(pi))
        0.45704579946446573615,  # sqrt(42) / (8 * sqrt(pi))
        0.37317633259011539140,  # sqrt(7) / (4 * sqrt(pi))
        0.45704579946446573615,  # sqrt(42) / (8 * sqrt(pi))
        1.44530572132027702772,  # sqrt(105) / (4 * sqrt(pi))
        0.59004358992664351035,  # sqrt(70) / (8 * sqrt(pi))
    ),
    # 4
    (
        2.50334294179670453838,  # 3 * sqrt(35) / (4 * sqrt(pi))
        1.77013076977993053104,  # 3 * sqrt(70) / (8 * sqrt(pi))
        0.94617469575756001803,  # 3 * sqrt(5) / (4 * sqrt(pi))
        0.66904654355728916795,  # 3 * sqrt(10) / (8 * sqrt(pi))
        0.10578554691520430380,  # 3 / (16 * sqrt(pi))
        0.66904654355728916795,  # 3 * sqrt(10) / (8 * sqrt(pi))
        0.47308734787878000902,  # 3 * sqrt(5) / (8 * sqrt(pi))
        1.77013076977993053104,  # 3 * sqrt(70) / (8 * sqrt(pi))
        0.62583573544917613459,  # 3 * sqrt(35) / (16 * sqrt(pi))
    ),
    # 5
    (
        0.65638205684017010284,  # 3 * sqrt(154) / (32 * sqrt(pi))
        8.30264925952416511616,  # 3 * sqrt(385) / (4 * sqrt(pi))
        0.48923829943525038767,  # sqrt(770) / (32 * sqrt(pi))
        4.79353678497332375474,  # sqrt(1155) / (4 * sqrt(pi))
        0.45294665119569692130,  # sqrt(165) / (16 * sqrt(pi))
        0.11695032245342359643,  # sqrt(11) / (16 * sqrt(pi))
        0.45294665119569692130,  # sqrt(165) / (16 * sqrt(pi))
        2.39676839248666187737,  # sqrt(1155) / (8 * sqrt(pi))
        0.48923829943525038767,  # sqrt(770) / (32 * sqrt(pi))
        2.07566231488104127904,  # 3 * sqrt(385) / (16 * sqrt(pi))
        0.65638205684017010284,  # 3 * sqrt(154) / (32 * sqrt(pi))
    ),
    # 6
    (
        1.36636821038382864392,  # sqrt(6006) / (32 * sqrt(pi))
        2.36661916223175203180,  # 3 * sqrt(2002) / (32 * sqrt(pi))
        2.01825960291489663696,  # 3 * sqrt(91) / (8 * sqrt(pi))
        0.92120525951492349912,  # sqrt(2730) / (32 * sqrt(pi))
        0.92120525951492349912,  # sqrt(2730) / (32 * sqrt(pi))
        0.58262136251873138879,  # sqrt(273) / (16 * sqrt(pi))
        0.06356920226762842593,  # sqrt(13) / (32 * sqrt(pi))
        0.58262136251873138879,  # sqrt(273) / (16 * sqrt(pi))
        0.46060262975746174956,  # sqrt(2730) / (64 * sqrt(pi))
        0.92120525951492349912,  # sqrt(2730) / (32 * sqrt(pi))
        0.50456490072872415924,  # 3 * sqrt(91) / (32 * sqrt(pi))
        2.36661916223175203180,  # 3 * sqrt(2002) / (32 * sqrt(pi))
        0.68318410519191432196,  # sqrt(6006) / (64 * sqrt(pi))
    ),
    (
        0.70716273252459617822,  # 3 * sqrt(715) / (64 * sqrt(pi))
        5.29192132360380044402,  # 3 * sqrt(10010) / (32 * sqrt(pi))
        0.51891557872026031976,  # 3 * sqrt(385) / (64 * sqrt(pi))
        4.15132462976208255808,  # 3 * sqrt(385) / (8 * sqrt(pi))
        0.15645893386229403365,  # 3 * sqrt(35) / (64 * sqrt(pi))
        0.44253269244498263276,  # 3 * sqrt(70) / (32 * sqrt(pi))
        0.09033160758251731423,  # sqrt(105) / (64 * sqrt(pi))
        0.06828427691200494191,  # sqrt(15) / (32 * sqrt(pi))
        0.09033160758251731423,  # sqrt(105) / (64 * sqrt(pi))
        0.22126634622249131638,  # 3 * sqrt(70) / (64 * sqrt(pi))
        0.15645893386229403365,  # 3 * sqrt(35) / (64 * sqrt(pi))
        1.03783115744052063952,  # 3 * sqrt(385) / (32 * sqrt(pi))
        0.51891557872026031976,  # 3 * sqrt(385) / (64 * sqrt(pi))
        2.64596066180190022201,  # 3 * sqrt(10010) / (64 * sqrt(pi))
        0.70716273252459617822,  # 3 * sqrt(715) / (64 * sqrt(pi))
    ),
)

def generate_sh_funcs(n_degrees: int) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    The encoded xyz coordinates should be normalized to [-1, 1] before passing to this function.
    """
    assert isinstance(n_degrees, int), "n_degrees must be a integer."
    assert 1 <= n_degrees <= 8, "only first 1 ~ 8 degree of spherical harmonics are supported."

    def sh_funcs(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        tensor_out = []
        tensor_out.append(torch.full_like(x, C[0][0]))
        if n_degrees <= 1:
            return tensor_out

        tensor_out.append(C[1][0] * y)
        tensor_out.append(C[1][1] * z)
        tensor_out.append(C[1][2] * x)
        if n_degrees <= 2:
            return tensor_out

        x2, y2, z2, xy, xz, yz = x * x, y * y, z * z, x * y, x * z, y * z
        tensor_out.append(C[2][0] * xy)
        tensor_out.append(C[2][1] * yz)
        tensor_out.append(C[2][2] * (3.0 * z2 - 1.0))
        tensor_out.append(C[2][3] * xz)
        tensor_out.append(C[2][4] * (x2 - y2))
        if n_degrees <= 3:
            return tensor_out

        tensor_out.append(C[3][0] * (3.0 * x2 - y2) * y)
        tensor_out.append(C[3][1] * xy * z)
        tensor_out.append(C[3][2] * (5.0 * z2 - 1.0) * y)
        tensor_out.append(C[3][3] * (5.0 * z2 - 3.0) * z)
        tensor_out.append(C[3][4] * (5.0 * z2 - 1.0) * x)
        tensor_out.append(C[3][5] * (x2 - y2) * z)
        tensor_out.append(C[3][6] * (x2 - 3.0 * y2) * x)
        if n_degrees <= 4:
            return tensor_out

        x4, y4, z4 = x2 * x2, y2 * y2, z2 * z2, 
        tensor_out.append(C[4][0] * xy * (x2 - y2))
        tensor_out.append(C[4][1] * yz * (3.0 * x2 - y2))
        tensor_out.append(C[4][2] * xy * (7.0 * z2 - 1.0))
        tensor_out.append(C[4][3] * yz * (7.0 * z2 - 3.0))
        tensor_out.append(C[4][4] * (35.0 * z4 - 30.0 * z2 + 3.0))
        tensor_out.append(C[4][5] * xz * (7.0 * z2 - 3.0))
        tensor_out.append(C[4][6] * (x2 - y2) * (7.0 * z2 - 1.0))
        tensor_out.append(C[4][7] * xz * (x2 - 3.0 * y2))
        tensor_out.append(C[4][8] * (x4 - 6.0 * x2 * y2 + y4))
        if n_degrees <= 5:
            return tensor_out

        xyz = x * y * z
        tensor_out.append(C[5][ 0] * y * (5.0 * x4 - 10.0 * x2 * y2 + y4))
        tensor_out.append(C[5][ 1] * xyz * (x2 - y2))
        tensor_out.append(C[5][ 2] * y * (3.0 * x2 - y2) * (9.0 * z2 - 1.0))
        tensor_out.append(C[5][ 3] * xyz * (3.0 * z2 - 1.0))
        tensor_out.append(C[5][ 4] * y * (21.0 * z4 - 14.0 * z2 + 1.0))
        tensor_out.append(C[5][ 5] * z * (63.0 * z4 - 70.0 * z2 + 15.0))
        tensor_out.append(C[5][ 6] * x * (21.0 * z4 - 14.0 * z2 + 1.0))
        tensor_out.append(C[5][ 7] * z * (x2 - y2) * (3.0 * z2 - 1.0))
        tensor_out.append(C[5][ 8] * x * (x2 - 3.0 * y2) * (9.0 * z2 - 1.0))
        tensor_out.append(C[5][ 9] * z * (x4 - 6.0 * x2 * y2 + y4))
        tensor_out.append(C[5][10] * x * (x4 - 10.0 * x2 * y2 + 5.0 * y4))
        if n_degrees <= 6:
            return tensor_out

        x6, y6, z6 = x2 * x4, y2 * y4, z2 * z4
        tensor_out.append(C[6][ 0] * xy * (3.0 * x4 - 10.0 * x2 * y2 + 3.0 * y4))
        tensor_out.append(C[6][ 1] * yz * (5.0 * x4 - 10.0 * x2 * y2 + y4))
        tensor_out.append(C[6][ 2] * xy * (x2 - y2) * (11.0 * z2 - 1.0))
        tensor_out.append(C[6][ 3] * yz * (3.0 * x2 - y2) * (11.0 * z2 - 3.0))
        tensor_out.append(C[6][ 4] * xy * (33.0 * z4 - 18.0 * z2 + 1.0))
        tensor_out.append(C[6][ 5] * yz * (33.0 * z4 - 30.0 * z2 + 5.0))
        tensor_out.append(C[6][ 6] * (231.0 * z6 - 315.0 * z4 + 105.0 * z2 - 5.0))
        tensor_out.append(C[6][ 7] * xz * (33.0 * z4 - 30.0 * z2 + 5.0))
        tensor_out.append(C[6][ 8] * (x2 - y2) * (33.0 * z4 - 18.0 * z2 + 1.0))
        tensor_out.append(C[6][ 9] * xz * (x2 - 3.0 * y2) * (11.0 * z2 - 3.0))
        tensor_out.append(C[6][10] * (x4 - 6.0 * x2 * y2 + y4) * (11.0 * z2 - 1.0))
        tensor_out.append(C[6][11] * xz * (x4 - 10.0 * x2 * y2 + 5.0 * y4))
        tensor_out.append(C[6][12] * (x6 - 15.0 * x4 * y2 + 15.0 * x2 * y4 - y6))
        if n_degrees <= 7:
            return tensor_out

        tensor_out.append(C[7][ 0] * y * (7.0 * x6 - 35.0 * x4 * y2 + 21.0 * x2 * y4 - y6))
        tensor_out.append(C[7][ 1] * xyz * (3.0 * x4 - 10.0 * x2 * y2 + 3.0 * y4))
        tensor_out.append(C[7][ 2] * y * (13.0 * z2 - 1.0) * (5.0 * x4 - 10.0 * x2 * y2 + y4))
        tensor_out.append(C[7][ 3] * xyz * (13.0 * z2 - 3.0) * (x2 - y2))
        tensor_out.append(C[7][ 4] * y * (143.0 * z4 - 66.0 * z2 + 3.0) * (3.0 * x2 - y2))
        tensor_out.append(C[7][ 5] * xyz * (143.0 * z4 - 110.0 * z2 + 15.0))
        tensor_out.append(C[7][ 6] * y * (429.0 * z6 - 495.0 * z4 + 135.0 * z2 - 5.0))
        tensor_out.append(C[7][ 7] * z * (429.0 * z6 - 693.0 * z4 + 315.0 * z2 - 35.0))
        tensor_out.append(C[7][ 8] * x * (429.0 * z6 - 495.0 * z4 + 135.0 * z2 - 5.0))
        tensor_out.append(C[7][ 9] * z * (143.0 * z4 - 110.0 * z2 + 15.0) * (x2 - y2))
        tensor_out.append(C[7][10] * x * (143.0 * z4 - 66.0 * z2 + 3.0) * (x2 - 3.0 * y4))
        tensor_out.append(C[7][11] * z * (13.0 * z2 - 3.0) * (x4 - 6.0 * x2 * y2 + y4))
        tensor_out.append(C[7][12] * x * (13.0 * z2 - 1.0) * (x4 - 10.0 * x2 * y2 - 5.0 * y4))
        tensor_out.append(C[7][13] * z * (x6 - 15.0 * x4 * y2 + 15.0 * x2 * y4 - y6))
        tensor_out.append(C[7][14] * x * (x6 - 21.0 * x4 * y2 + 35.0 * x2 * y4 - 7.0 * y6))
        if n_degrees <= 8:
            return tensor_out

    return sh_funcs
