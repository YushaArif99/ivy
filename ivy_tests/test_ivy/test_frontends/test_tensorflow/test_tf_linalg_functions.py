# global
import ivy
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers

@st.composite
def dtype_matrix_n_tolr(draw):

    dtype_and_x = draw(
        helpers.dtype_and_values(
            available_dtypes=(ivy.float32, ivy.float64),
            min_num_dims=2,
            max_num_dims=2,
            max_dim_size=1,
            max_value=1.0,
        )
    )

    dt, matrix = dtype_and_x
    size = draw(st.integers(1, 10))
    if size % 2 == 0:
        tolr = draw(
            helpers.list_of_length(
                x=st.floats(
                    width=16,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False
                ), length=1
            )
        )[0]
    else:
        tolr = None

    return (dt, matrix, tolr)


@given(
    dtype_and_x_and_tolr=dtype_matrix_n_tolr(),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.matrix_rank"
    ),
    native_array=st.booleans(),
)
def test_matrix_rank(
    dtype_and_x_and_tolr,
    as_variable,
    num_positional_args,
    native_array,
    fw
):
    input_dtype, x, tolr = dtype_and_x_and_tolr
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_name="linalg.matrix_rank",
        a=np.asarray(x, dtype=input_dtype),
        tol=tolr
    )
