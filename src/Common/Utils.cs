using System;
using System.Collections.Generic;
using System.IO;
using Zcu.Convsharp.Logger;
using Zcu.Convsharp.Model;

namespace Zcu.Convsharp.Common
{
    /// <summary>
    /// Library class for common methods which are used
    /// on different parts of program
    /// </summary>
    public static class Utils
    {
        /// <summary>
        /// Instance of 
        /// generator which we are using due to possibility
        /// to set same seed for each start of program. If we will have same seed
        /// we will get same results everytime
        /// </summary>
        private static Random randomGenerator = new Random();
        /// <summary>
        /// Method which converts list of labels into categorical array
        /// e.g. data = {1, 3, 0, 1, 2, 4, 3}, numberOfClass = 5
        /// output = {[0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [1, 0, 0, 0, 0], ...}
        /// </summary>
        /// <param name="data">1D byte array of labels</param>
        /// <param name="numberOfClass">number of class which will classify</param>
        /// <returns></returns>
        public static Tuple<double[][], Dictionary<int, uint>> ToCategorical(int[] data, uint numberOfClass, Dictionary<int, uint> classes = null)
        {
            double[][] categoricalArray = new double[data.Length][];
            if (classes == null)
                classes = new Dictionary<int, uint>();
            uint index = 0;
            for (int i = 0; i < data.Length; i++)
            {
                double[] row = new double[numberOfClass];
                if (!classes.ContainsKey(data[i]))
                {
                    classes.Add(data[i], index++);
                }
                row[classes[data[i]]] = 1f;
                categoricalArray[i] = row;
            }
            return Tuple.Create(categoricalArray, classes);
        }

        /// <summary>
        /// Throw expception with error message
        /// </summary>
        /// <param name="msg">error message which will be logged</param>
        public static void ThrowException(string msg)
        {
            Log.Error(msg);
            throw new Exception(msg);
        }

        /// <summary>
        /// Test if is dimension changed with parameters which will
        /// be compared with input dimension
        /// </summary>
        /// <param name="inputDimension">input dimension</param>
        /// <param name="currImageCount">image count parameter</param>
        /// <param name="currDepth">depth parameter</param>
        /// <param name="currWidth">width parameter</param>
        /// <param name="currHeight">height parameter</param>
        /// <returns></returns>
        public static bool IsDimensionChanged(Dimension inputDimension, int currImageCount, int currDepth,
            int currWidth, int currHeight)
        {
            if (inputDimension == null || !inputDimension.IsSame(currImageCount, currDepth,
                 currWidth, currHeight))
            {
                inputDimension = new Dimension();
                inputDimension.imageCount = currImageCount;
                inputDimension.depth = currDepth;
                inputDimension.width = currWidth;
                inputDimension.height = currHeight;
                return true;
            }
            return false;
        }

        /// <summary>
        /// Generate random number from uniform distribution
        /// </summary>
        /// <returns></returns>
        public static double GetRandomValueUniform()
        {
            return (randomGenerator.NextDouble() - 0.5);
        }

        /// <summary>
        /// Method for simple initialization of 4D array
        /// </summary>
        /// <param name="v0">size of first dimension</param>
        /// <param name="v1">size of second dimension</param>
        /// <param name="v2">size of third dimension</param>
        /// <param name="v3">size of fourth dimension</param>
        /// <returns>initialized array</returns>
        public static double[][][][] Init4dArr(int v0, int v1, int v2, int v3)
        {
            double[][][][] x = new double[v0][][][];
            for (int i = 0; i < v0; i++)
            {
                x[i] = new double[v1][][];
                for (int j = 0; j < v1; j++)
                {
                    x[i][j] = new double[v2][];
                    for (int k = 0; k < v2; k++)
                    {
                        x[i][j][k] = new double[v3];
                    }
                }
            }
            return x;
        }

        /// <summary>
        /// Method which load sequentional model which was serialized
        /// and saved to disk.
        /// </summary>
        /// <param name="pathToModel">Path to file with saved model</param>
        /// <returns>Loaded model</returns>
        public static SequentialModel LoadModel(string pathToModel)
        {
            SequentialModel model = null;

            try
            {
                FileStream stream = File.OpenRead(pathToModel);
                var formatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                model = (SequentialModel)formatter.Deserialize(stream);
                stream.Close();
            }
            catch (Exception ex)
            {
                ThrowException("Loading model failed with following error message " + ex.Message);
            }

            return model;
        }

        /// <summary>
        /// Method for simple initialization of 2D array
        /// </summary>
        /// <param name="v0">size of first dimension</param>
        /// <param name="v1">size of second dimension</param>
        /// <returns>initialized array</returns>
        public static double[][] Init2dArr(int v0, int v1)
        {
            double[][] x = new double[v0][];
            for (int i = 0; i < v0; i++)
            {
                x[i] = new double[v1];
            }
            return x;
        }

        /// <summary>
        /// Method for simple initialization of 3D array
        /// </summary>
        /// <param name="v0">size of first dimension</param>
        /// <param name="v1">size of second dimension</param>
        /// <param name="v2">size of third dimension</param>
        /// <returns>initialized array</returns>
        public static double[][][] Init3dArr(int v0, int v1, int v2)
        {
            double[][][] x = new double[v0][][];
            for (int i = 0; i < v0; i++)
            {
                x[i] = new double[v1][];
                for (int j = 0; j < v2; j++)
                    x[i][j] = new double[v2];
            }
            return x;
        }

        /// <summary>
        /// Generate random integer between FROM - TO
        /// </summary>
        /// <param name="from">first/start border</param>
        /// <param name="to">second/end border</param>
        /// <returns>random integer</returns>
        public static int GetRandomInt(int from, int to)
        {
            int val = (int)(randomGenerator.NextDouble() * (to - from)) + from;
            return val;
        }

        /// <summary>
        /// Generate random double between FROM - TO
        /// </summary>
        /// <returns>random integer</returns>
        public static double GetRandomDouble()
        {
            return randomGenerator.NextDouble();
        }

        /// <summary>
        /// Method for simple initialization of 4D array
        /// </summary>
        /// <param name="v0">size of first dimension</param>
        /// <param name="v1">size of second dimension</param>
        /// <param name="v2">size of third dimension</param>
        /// <param name="v3">size of fourth dimension</param>
        /// <param name="v4">size of fifth dimension</param>
        /// <returns>initialized array</returns>
        public static int[][][][][] InitInt5dArr(int v0, int v1, int v2, int v3, int v4)
        {
            int[][][][][] x = new int[v0][][][][];
            for (int i = 0; i < v0; i++)
            {
                x[i] = new int[v1][][][];
                for (int j = 0; j < v1; j++)
                {
                    x[i][j] = new int[v2][][];
                    for (int k = 0; k < v2; k++)
                    {
                        x[i][j][k] = new int[v3][];
                        for (int l = 0; l < v3; l++)
                        {
                            x[i][j][k][l] = new int[v4];
                        }
                    }
                }
            }
            return x;
        }
    }
}