// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/utils/type_erased_matrix.hpp>

// Other includes
#include <El.hpp> // IWYU pragma: export

namespace
{
template <typename SrcT, typename TgtT>
struct TypePair
{
  using src_type = SrcT;
  using tgt_type = TgtT;
};
}// namespace <Anon>

TEMPLATE_PRODUCT_TEST_CASE(
  "Testing type-erase Matrix","[type-erase][la][utilities]",
  (TypePair),
  ((int, float), (int, double),
   (float, int), (float, double),
   (double, int), (double,float)))
{
  using src_type = typename TestType::src_type;
  using tgt_type = typename TestType::tgt_type;

  GIVEN("A type-erased matrix")
  {
    auto mat = lbann::utils::create_type_erased_matrix<src_type>();

    THEN ("the internal matrix has the correct storage type")
    {
      REQUIRE_NOTHROW(mat->template get<src_type>());
      REQUIRE_THROWS_AS(mat->template get<tgt_type>(),
                        lbann::utils::bad_any_cast);

      auto&& internal_mat = mat->template get<src_type>();
      REQUIRE(internal_mat.Height() == 0);
      REQUIRE(internal_mat.Width() == 0);
    }

    WHEN ("The matrix is resized")
    {
      REQUIRE_NOTHROW(mat->template get<src_type>().Resize(10,12));

      THEN ("The change is reflected in the internal matrix.")
      {
        auto&& internal_mat = mat->template get<src_type>();
        REQUIRE(internal_mat.Height() == 10);
        REQUIRE(internal_mat.Width() == 12);
      }
      AND_WHEN ("The matrix is changed")
      {
        REQUIRE_NOTHROW(mat->template emplace<tgt_type>(14,10));

        THEN ("The internal matrix has the right type and size")
        {
          REQUIRE_NOTHROW(mat->template get<tgt_type>());
          REQUIRE_THROWS_AS(mat->template get<src_type>(),
                            lbann::utils::bad_any_cast);

          REQUIRE(mat->template get<tgt_type>().Height() == 14);
          REQUIRE(mat->template get<tgt_type>().Width() == 10);
        }
      }
    }
  }

  GIVEN("A matrix of a given type")
  {
    El::Matrix<src_type> mat(10,12);
    mat(1,1) = src_type(13);

    WHEN("A type-erased matrix is constructed by copying it")
    {
      lbann::utils::type_erased_matrix erased_mat(mat);
      THEN("The type-erased matrix is a copy")
      {
        REQUIRE(erased_mat.template get<src_type>().Height() == 10);
        REQUIRE(erased_mat.template get<src_type>().Width() == 12);
        REQUIRE(
          erased_mat.template get<src_type>().operator()(1,1) == mat(1,1));

        AND_WHEN("The original matrix is resized")
        {
          mat.Resize(5,5);
          THEN("The type-erased matrix is unaffected.")
          {
            REQUIRE(erased_mat.template get<src_type>().Height() == 10);
            REQUIRE(erased_mat.template get<src_type>().Width() == 12);
          }
        }
      }
    }

    WHEN("A type-erased matrix is constructed by moving it")
    {
      lbann::utils::type_erased_matrix erased_mat(std::move(mat));
      THEN("The type-erased matrix is sized correctly and has good values")
      {
        REQUIRE(erased_mat.template get<src_type>().Height() == 10);
        REQUIRE(erased_mat.template get<src_type>().Width() == 12);
        REQUIRE(
          erased_mat.template get<src_type>().operator()(1,1) == src_type(13));

        AND_WHEN("The original matrix is resized")
        {
          mat.Resize(5,5);
          THEN("The type-erased matrix is unaffected.")
          {
            REQUIRE(erased_mat.template get<src_type>().Height() == 10);
            REQUIRE(erased_mat.template get<src_type>().Width() == 12);
          }
        }
      }
    }
  }
}
