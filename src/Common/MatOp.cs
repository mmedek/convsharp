using System;

namespace Zcu.Convsharp.Common
{
    /// <summary>
    /// Class for simple matrix operations
    /// </summary>
    public static class MatOp
    {

        /// <summary>
        /// Compute square root for each item in matrix
        /// </summary>
        /// <param name="x">input matrix</param>
        /// <returns>matrix with square rooted items</returns>
        public static double[][][][] Sqrt(double[][][][] x)
        {
            int lx0 = x.Length;
            int lx1 = x[0].Length;
            int lx2 = x[0][0].Length;
            int lx3 = x[0][0][0].Length;
            // init array for results
            double[][][][] resultMatrix = Utils.Init4dArr(lx0, lx1, lx2, lx3);
            for (int i = 0; i < lx0; i++)
            {
                for (int j = 0; j < lx1; j++)
                {
                    for (int k = 0; k < lx2; k++)
                    {
                        for (int l = 0; l < lx3; l++)
                            resultMatrix[i][j][k][l] = Math.Sqrt(x[i][j][k][l]);
                    }
                }
            }
            return resultMatrix;
        }

        /// <summary>
        /// Method for transposing input array, we can
        /// transpose only part of the input matrix if we
        /// will use start and end indexes if not
        /// all input matrix will be transposed.
        /// We expecting matrix in shape (X, 1, 1, Y)
        /// </summary>
        /// <param name="x">input matrix (X, 1, 1, Y)</param>
        /// <param name="startIndex">first value of submatrix</param>
        /// <param name="endIndex">last value of submatrix</param>
        /// <returns>transposed array (Y, 1, 1, X)</returns>
        public static double[][][][] Transpose(double[][][][] x, int startIndex = 0, int endIndex = 0)
        {
            int width = x[0][0][0].Length;
            int numSamples;
            double[][][][] transposed;
            // we will transpose only part of array if startIndex and endIndex
            // is set
            if (startIndex == endIndex)
            {
                numSamples = x.Length;
                endIndex = numSamples;
            }
            else
            {
                numSamples = endIndex - startIndex;
            }
            // init array for results
            transposed = Utils.Init4dArr(width, 1, 1, numSamples);
            // transpose
            int index;
            for (int i = 0; i < width; i++)
            {
                index = 0;
                for (int j = startIndex; j < endIndex; j++)
                {
                    transposed[i][0][0][index] = x[index][0][0][i];
                    index++;
                }
            }
            return transposed;
        }

        /// <summary>
        /// Compute mean of matrix, expecting matrix in shape
        /// (X, 1, 1, Y)
        /// </summary>
        /// <param name="input">input array (X, 1, 1, Y)</param>
        /// <returns>array of means for each row</returns>
        public static double[] Mean(double[][][][] input)
        {
            int height = input.Length;
            int width = input[0][0][0].Length;
            double[] b = new double[width];
            double sum = 0;
            for (int i = 0; i < width; i++)
            {
                sum = 0;
                for (int j = 0; j < height; j++)
                {
                    sum += input[j][0][0][i];
                }
                b[i] = sum / height;
            }
            return b;
        }

        #region Matrix element wise operations
        /// <summary>
        /// Method for Cwise multiplication of two input arrays.
        /// Cwise multiplication is simple element wise multiplication
        /// e.g. [1,2] * [3,4] = [3,8]
        /// </summary>
        /// <param name="x">first input array</param>
        /// <param name="y">second input array</param>
        /// <returns>resulting array</returns>
        public static double[][][][] Cwise(double[][][][] x, double[][][][] y)
        {
            int lx0 = x.Length;
            int lx1 = x[0].Length;
            int lx2 = x[0][0].Length;
            int lx3 = x[0][0][0].Length;
            int ly0 = y.Length;
            int ly1 = y[0].Length;
            int ly2 = y[0][0].Length;
            int ly3 = y[0][0][0].Length;
            // check dimension
            if (lx0 != ly0 || lx1 != ly1 || lx2 != ly2 || lx3 != ly3)
            {
                string msg = "Invalid dimension of matrixes during CWISE product.";
                Utils.ThrowException(msg);
            }
            // init array for results
            double[][][][] resMatrix = Utils.Init4dArr(lx0, lx1, lx2, lx3);
            for (int i = 0; i < lx0; i++)
            {
                for (int j = 0; j < lx1; j++)
                {
                    for (int k = 0; k < lx2; k++)
                    {
                        for (int l = 0; l < lx3; l++)
                            resMatrix[i][j][k][l] = x[i][j][k][l] * y[i][j][k][l];
                    }
                }
            }
            return resMatrix;
        }

        /// <summary>
        /// Method for Mwise division of two input arrays of shape.
        /// Mwise division is simple element wise division
        /// e.g. [4,4] / [2,2] = [2,2]
        /// </summary>
        /// <param name="x">first input array</param>
        /// <param name="y">second input array</param>
        /// <returns>resulting array</returns>
        public static double[][][][] Mwise(double[][][][] x, double[][][][] y)
        {
            int lx0 = x.Length;
            int lx1 = x[0].Length;
            int lx2 = x[0][0].Length;
            int lx3 = x[0][0][0].Length;
            int ly0 = y.Length;
            int ly1 = y[0].Length;
            int ly2 = y[0][0].Length;
            int ly3 = y[0][0][0].Length;
            // check dimension
            if (lx0 != ly0 || lx1 != ly1 || lx2 != ly2 || lx3 != ly3)
            {
                string msg = "Invalid dimension of matrixes during MWISE product.";
                Utils.ThrowException(msg);
            }
            // init array for results
            double[][][][] resMatrix = Utils.Init4dArr(lx0, lx1, lx2, lx3);
            for (int i = 0; i < lx0; i++)
            {
                for (int j = 0; j < lx1; j++)
                {
                    for (int k = 0; k < lx2; k++)
                    {
                        for (int l = 0; l < lx3; l++)
                            resMatrix[i][j][k][l] = x[i][j][k][l] / y[i][j][k][l];
                    }
                }
            }
            return resMatrix;
        }
        #endregion Matrix element wise operations

        #region Matrix X constant operations
        /// <summary>
        /// Add constant number to each item in matrix
        /// </summary>
        /// <param name="x">input matrix</param>
        /// <param name="constant">input number which will be add to each item
        /// in matrix</param>
        /// <returns>matrix with added constant</returns>
        public static double[][][][] AddConst(double[][][][] x, double constant)
        {
            int lx0 = x.Length;
            int lx1 = x[0].Length;
            int lx2 = x[0][0].Length;
            int lx3 = x[0][0][0].Length;
            // init array for results
            double[][][][] resultMatrix = Utils.Init4dArr(lx0, lx1, lx2, lx3);
            for (int i = 0; i < lx0; i++)
            {
                for (int j = 0; j < lx1; j++)
                {
                    for (int k = 0; k < lx2; k++)
                    {
                        for (int l = 0; l < lx3; l++)
                            resultMatrix[i][j][k][l] = x[i][j][k][l] + constant;
                    }
                }
            }
            return resultMatrix;
        }

        /// <summary>
        /// Multiply each element of input matrix with constant variable
        /// </summary>
        /// <param name="x">input matrix</param>
        /// <param name="cons">constant variable</param>
        /// <returns>resulting array</returns>
        public static double[][][][] MultiplyByConst(double[][][][] x, double cons)
        {
            int lx0 = x.Length;
            int lx1 = x[0].Length;
            int lx2 = x[0][0].Length;
            int lx3 = x[0][0][0].Length;
            // init array for results
            double[][][][] resultMatrix = Utils.Init4dArr(lx0, lx1, lx2, lx3);
            for (int i = 0; i < lx0; i++)
            {
                for (int j = 0; j < lx1; j++)
                {
                    for (int k = 0; k < lx2; k++)
                    {
                        for (int l = 0; l < lx3; l++)
                            resultMatrix[i][j][k][l] = x[i][j][k][l] * cons;
                    }
                }
            }
            return resultMatrix;
        }

        /// <summary>
        /// Divide each element of input matrix with constant variable
        /// </summary>
        /// <param name="x">input matrix</param>
        /// <param name="cons">constant variable</param>
        /// <returns>resulting array</returns>
        public static double[][][][] DivideByConst(double[][][][] x, double cons)
        {
            int lx0 = x.Length;
            int lx1 = x[0].Length;
            int lx2 = x[0][0].Length;
            int lx3 = x[0][0][0].Length;
            // init array for results
            double[][][][] resultMatrix = Utils.Init4dArr(lx0, lx1, lx2, lx3);
            for (int i = 0; i < lx0; i++)
            {
                for (int j = 0; j < lx1; j++)
                {
                    for (int k = 0; k < lx2; k++)
                    {
                        for (int l = 0; l < lx3; l++)
                            resultMatrix[i][j][k][l] = x[i][j][k][l] / cons;
                    }
                }
            }
            return resultMatrix;
        }

        #endregion Matrix X constant operations

        #region Matrix X matrix operations
        /// <summary>
        /// Subtract two matrixes. E.g. [4,4] - [2,2] = [2,2]
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public static double[][][][] Substract(double[][][][] x, double[][][][] y)
        {
            int lx0 = x.Length;
            int lx1 = x[0].Length;
            int lx2 = x[0][0].Length;
            int lx3 = x[0][0][0].Length;
            int ly0 = y.Length;
            int ly1 = y[0].Length;
            int ly2 = y[0][0].Length;
            int ly3 = y[0][0][0].Length;
            // check dimension
            if (lx0 != ly0 || lx1 != ly1 || lx2 != ly2 || lx3 != ly3)
            {
                string msg = "Invalid dimension of matrixes during substraction.";
                Utils.ThrowException(msg);
            }
            // init array for results
            double[][][][] resultMatrix = Utils.Init4dArr(lx0, lx1, lx2, lx3);

            for (int i = 0; i < lx0; i++)
            {
                for (int j = 0; j < lx1; j++)
                {
                    for (int k = 0; k < lx2; k++)
                    {
                        for (int l = 0; l < lx3; l++)
                            resultMatrix[i][j][k][l] = x[i][j][k][l] - y[i][j][k][l];
                    }
                }
            }
            return resultMatrix;
        }

        /// <summary>
        /// Add two matrixes. E.g. [4,4] * [2,2] = [6,6]
        /// </summary>
        /// <param name="x">first matrix</param>
        /// <param name="y">second matrix</param>
        /// <returns>output matrix</returns>
        public static double[][][][] Add(double[][][][] x, double[][][][] y)
        {
            int lx0 = x.Length;
            int lx1 = x[0].Length;
            int lx2 = x[0][0].Length;
            int lx3 = x[0][0][0].Length;
            int ly0 = y.Length;
            int ly1 = y[0].Length;
            int ly2 = y[0][0].Length;
            int ly3 = y[0][0][0].Length;
            // check dimension
            if (lx0 != ly0 || lx1 != ly1 || lx2 != ly2 || lx3 != ly3)
            {
                string msg = "Invalid dimension of matrixes during Add operation.";
                Utils.ThrowException(msg);
            }
            // init array for results
            double[][][][] resMatrix = Utils.Init4dArr(lx0, lx1, lx2, lx3);
            for (int i = 0; i < lx0; i++)
            {
                for (int j = 0; j < lx1; j++)
                {
                    for (int k = 0; k < lx2; k++)
                    {
                        for (int l = 0; l < lx3; l++)
                            resMatrix[i][j][k][l] = x[i][j][k][l] + y[i][j][k][l];
                    }
                }
            }
            return resMatrix;
        }

        /// <summary>
        /// Method for computing dot product of two 4d arrays,
        /// we are expecting that arrays have shape (X, 1, 1, Y)
        /// before that operation
        /// </summary>
        /// <param name="x">first input array</param>
        /// <param name="y">second input array</param>
        /// <param name="divisor">if we want to normalize array with 
        /// some number we can add divisor which will be used
        /// for each number in result matrix</param>
        /// <returns>result matrix</returns>
        public static double[][][][] Dot(double[][][][] x, double[][][][] y, double divisor = 1)
        {
            int rowsX = x.Length;
            int colsX = x[0][0][0].Length;
            int rowsY = y.Length;
            int colsY = y[0][0][0].Length;
            // test dimension
            if (colsX != rowsY)
            {
                string msg = "Invalid dimension of matrix during computing dot product.";
                Utils.ThrowException(msg);
            }
            // multiplicator which is use for normalization
            double multiplicator = 1 / divisor;
            // init array for results
            double[][][][] results = Utils.Init4dArr(rowsX, 1, 1, colsY);
            // temporary variables
            double sum = 0;
            int index = 0;
            for (int j = 0; j < colsY; j++)
            {
                for (int i = 0; i < rowsX; i++)
                {
                    sum = 0;
                    for (int k = 0; k < colsX; k++)
                    {
                        sum += x[i][0][0][k] * y[k][0][0][index];
                    }
                    results[i][0][0][j] = sum * multiplicator;
                }
                index++;
            }
            return results;
        }
        #endregion Matrix X matrix operations
    }
}