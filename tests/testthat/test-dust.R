context("dust")

test_that("cpp11 setup is working", {
  expect_equal(cpp_add(1, 2), 3)
})
