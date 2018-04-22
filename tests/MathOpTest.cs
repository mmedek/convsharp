using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Zcu.Convsharp.Common;

namespace Zcu.Convsharp.UnitTests
{
    [TestClass]
    public class MathOpTest
    {
        [TestMethod]
        public void SqrtTest()
        {
            int lastDim = 4;

            double[][][][] expected = Utils.Init4dArr(1, 1, 1, lastDim);
            expected[0][0][0][0] = 4;
            expected[0][0][0][1] = 16;
            expected[0][0][0][2] = 64;
            expected[0][0][0][3] = 256;

            double[] target = new double[lastDim];
            target[0] = 2;
            target[1] = 4;
            target[2] = 8;
            target[3] = 16;

            expected = MatOp.Sqrt(expected);

            for (int i = 0; i < lastDim; i++)
            {
                Assert.AreEqual(target[i], expected[0][0][0][i]);
            }
        }

        [TestMethod]
        public void TransposeTest()
        {
            int x = 4;
            int y = 2;

            double[][][][] expected = Utils.Init4dArr(x, 1, 1, y);
            double[][][][] target = Utils.Init4dArr(y, 1, 1, x);

            expected = MatOp.Transpose(expected);

            Assert.AreEqual(target.Length, expected.Length);
            Assert.AreEqual(target[0][0][0].Length, expected[0][0][0].Length);
        }

        [TestMethod]
        public void MeanTest()
        {
            int lastDim = 3;

            double[][][][] temp = Utils.Init4dArr(3, 1, 1, lastDim);
            temp[0][0][0][0] = 3;
            temp[0][0][0][1] = 6;
            temp[0][0][0][2] = 9;

            temp[1][0][0][0] = 6;
            temp[1][0][0][1] = 9;
            temp[1][0][0][2] = 12;

            temp[2][0][0][0] = 9;
            temp[2][0][0][1] = 12;
            temp[2][0][0][2] = 15;

            double[] target = new double[lastDim];
            target[0] = 6;
            target[1] = 9;
            target[2] = 12;

            double[] expected = MatOp.Mean(temp);

            for (int i = 0; i < lastDim; i++)
            {
                Assert.AreEqual(target[i], expected[i]);
            }
        }

        #region Matrix element wise operations
        [TestMethod]
        public void CwiseTest()
        {
            int x = 2;
            int y = 2;

            double[][][][] a = Utils.Init4dArr(x, 1, 1, y);
            a[0][0][0][0] = 1;
            a[0][0][0][1] = 2;
            a[1][0][0][0] = 3;
            a[1][0][0][1] = 4;

            double[][][][] b = Utils.Init4dArr(x, 1, 1, y);
            b[0][0][0][0] = 4;
            b[0][0][0][1] = 2;
            b[1][0][0][0] = 3;
            b[1][0][0][1] = 2.25d;

            double[][][][] expected = MatOp.Cwise(a, b);

            double[][][][] target = Utils.Init4dArr(x, 1, 1, y);
            target[0][0][0][0] = 4;
            target[0][0][0][1] = 4;
            target[1][0][0][0] = 9;
            target[1][0][0][1] = 9;

            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    Assert.AreEqual(target[i][0][0][j], expected[i][0][0][j]);
                }
            }
        }

        [TestMethod]
        [ExpectedException(typeof(Exception))]
        public void CwiseExceptionTest()
        {
            int x = 2;
            int y = 3;

            double[][][][] a = Utils.Init4dArr(x, 1, 1, x);
            double[][][][] b = Utils.Init4dArr(y, 1, 1, x);

            MatOp.Cwise(a, b);
        }

        [TestMethod]
        public void MwiseTest()
        {
            int x = 2;
            int y = 2;

            double[][][][] a = Utils.Init4dArr(x, 1, 1, y);
            a[0][0][0][0] = 1;
            a[0][0][0][1] = 2;
            a[1][0][0][0] = 3;
            a[1][0][0][1] = 4;

            double[][][][] b = Utils.Init4dArr(x, 1, 1, y);
            b[0][0][0][0] = 1;
            b[0][0][0][1] = 2;
            b[1][0][0][0] = 3;
            b[1][0][0][1] = 4;

            double[][][][] expected = MatOp.Mwise(a, b);

            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    Assert.AreEqual(1d, expected[i][0][0][j]);
                }
            }
        }

        [TestMethod]
        [ExpectedException(typeof(Exception))]
        public void MwiseExceptionTest()
        {
            int x = 2;
            int y = 3;

            double[][][][] a = Utils.Init4dArr(x, 1, 1, x);
            double[][][][] b = Utils.Init4dArr(y, 1, 1, x);

            MatOp.Mwise(a, b);
        }
        #endregion Matrix element wise operations

        #region Matrix X constant operations
        [TestMethod]
        public void AddConstTest()
        {
            int lastDim = 2;
            int x = 5;

            double[][][][] a = Utils.Init4dArr(lastDim, 1, 1, lastDim);
            a[0][0][0][0] = 1;
            a[0][0][0][1] = 2;
            a[1][0][0][0] = 3;
            a[1][0][0][1] = 4;

            double[][][][] expected = MatOp.AddConst(a, x);

            for (int i = 0; i < lastDim; i++)
            {
                for (int j = 0; j < lastDim; j++)
                {
                    Assert.AreEqual((a[i][0][0][j] + 5), expected[i][0][0][j]);
                }
            }
        }

        [TestMethod]
        public void MultiplyByConstTest()
        {
            int lastDim = 2;
            int x = 5;

            double[][][][] a = Utils.Init4dArr(lastDim, 1, 1, lastDim);
            a[0][0][0][0] = 1;
            a[0][0][0][1] = 2;
            a[1][0][0][0] = 3;
            a[1][0][0][1] = 4;

            double[][][][] expected = MatOp.MultiplyByConst(a, x);

            for (int i = 0; i < lastDim; i++)
            {
                for (int j = 0; j < lastDim; j++)
                {
                    Assert.AreEqual((a[i][0][0][j] * x), expected[i][0][0][j]);
                }
            }
        }

        [TestMethod]
        public void DivideByConstTest()
        {
            int lastDim = 2;
            int x = 5;

            double[][][][] a = Utils.Init4dArr(lastDim, 1, 1, lastDim);
            a[0][0][0][0] = 1;
            a[0][0][0][1] = 2;
            a[1][0][0][0] = 3;
            a[1][0][0][1] = 4;

            double[][][][] expected = MatOp.DivideByConst(a, x);

            for (int i = 0; i < lastDim; i++)
            {
                for (int j = 0; j < lastDim; j++)
                {
                    Assert.AreEqual((a[i][0][0][j] / x), expected[i][0][0][j]);
                }
            }
        }
        #endregion Matrix X constant operations

        #region Matrix X matrix operations
        [TestMethod]
        public void SubstractTest()
        {
            int x = 2;
            int y = 3;

            double[][][][] a = Utils.Init4dArr(x, 1, 1, y);
            a[0][0][0][0] = 1;
            a[0][0][0][1] = 2;
            a[0][0][0][2] = 3;
            a[1][0][0][0] = 3;
            a[1][0][0][1] = 2;
            a[1][0][0][2] = 1;

            double[][][][] b = Utils.Init4dArr(x, 1, 1, y);
            b[0][0][0][0] = 1;
            b[0][0][0][1] = 2;
            b[0][0][0][2] = 3;
            b[1][0][0][0] = 3;
            b[1][0][0][1] = 2;
            b[1][0][0][2] = 1;

            double[][][][] expected = MatOp.Substract(a, b);

            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    Assert.AreEqual(0, expected[i][0][0][j]);
                }
            }
        }

        [TestMethod]
        [ExpectedException(typeof(Exception))]
        public void SubstractExceptionTest()
        {
            int x = 2;
            int y = 3;

            double[][][][] a = Utils.Init4dArr(x, 1, 1, x);
            double[][][][] b = Utils.Init4dArr(y, 1, 1, y);

            MatOp.Substract(a, b);
        }

        [TestMethod]
        public void AddTest()
        {
            int x = 2;
            int y = 3;

            double[][][][] a = Utils.Init4dArr(x, 1, 1, y);
            a[0][0][0][0] = 1;
            a[0][0][0][1] = 2;
            a[0][0][0][2] = 3;
            a[1][0][0][0] = 3;
            a[1][0][0][1] = 2;
            a[1][0][0][2] = 1;

            double[][][][] b = Utils.Init4dArr(x, 1, 1, y);
            b[0][0][0][0] = 3;
            b[0][0][0][1] = 2;
            b[0][0][0][2] = 1;
            b[1][0][0][0] = 1;
            b[1][0][0][1] = 2;
            b[1][0][0][2] = 3;

            double[][][][] expected = MatOp.Add(a, b);

            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    Assert.AreEqual(4, expected[i][0][0][j]);
                }
            }
        }

        [TestMethod]
        [ExpectedException(typeof(Exception))]
        public void AddExceptionTest()
        {
            int x = 2;
            int y = 3;

            double[][][][] a = Utils.Init4dArr(x, 1, 1, x);
            double[][][][] b = Utils.Init4dArr(y, 1, 1, y);

            MatOp.Add(a, b);
        }

        [TestMethod]
        public void DotTest()
        {
            int x = 2;
            int y = 3;

            double[][][][] a = Utils.Init4dArr(x, 1, 1, y);
            a[0][0][0][0] = 1;
            a[0][0][0][1] = 2;
            a[0][0][0][2] = 3;
            a[1][0][0][0] = 4;
            a[1][0][0][1] = 5;
            a[1][0][0][2] = 6;

            double[][][][] b = Utils.Init4dArr(y, 1, 1, x);
            b[0][0][0][0] = 1;
            b[0][0][0][1] = 2;
            b[1][0][0][0] = 3;
            b[1][0][0][1] = 4;
            b[2][0][0][0] = 5;
            b[2][0][0][1] = 6;

            double[][][][] target = Utils.Init4dArr(x, 1, 1, x);
            target[0][0][0][0] = 11;
            target[0][0][0][1] = 14;
            target[1][0][0][0] = 24.5d;
            target[1][0][0][1] = 32;

            double[][][][] expected = MatOp.Dot(a, b, divisor: 2);

            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < x; j++)
                {
                    Assert.AreEqual(target[i][0][0][j], expected[i][0][0][j]);
                }
            }

            Assert.AreEqual(target.Length, expected.Length);
            Assert.AreEqual(target[0][0][0].Length, expected[0][0][0].Length);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception))]
        public void DotExceptionTest()
        {
            int x = 2;
            int y = 3;

            double[][][][] a = Utils.Init4dArr(x, 1, 1, y);
            double[][][][] b = Utils.Init4dArr(x, 1, 1, y);

            MatOp.Dot(a, b);
        }
        #endregion Matrix X matrix operations
    }
}