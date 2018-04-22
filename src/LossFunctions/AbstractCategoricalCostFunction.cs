using System;
using System.Linq;
using Zcu.Convsharp.Common;

namespace Zcu.Convsharp.CostFunctions
{
    /// <summary>
    /// Class which contains method for computing
    /// accuracy for categorical data.
    /// It can be used in categorical cross entropy
    /// loss or in some loss functions implemented
    /// in the future.
    /// </summary>
    [Serializable]
    public abstract class AbstractCategoricalCostFunction : AbstractCostFunction
    {
        public override double ComputeAccuracy(double[][][][] currentInput, double[][] trainLabels)
        {
            int okCounter = 0;
            int maxIndex = 0;
           
            for (int i = 0; i < currentInput.Length; i++)
            {
                maxIndex = GetResult(currentInput[i]);
                int res = Convert.ToInt32(trainLabels[i][maxIndex]);
                if (res == 1)
                {
                    okCounter++;
                }
            }

            double correct = Convert.ToDouble(okCounter);
            double all = Convert.ToDouble(currentInput.Length);
            double acc = correct / all;
            return acc;
        }

        public override int GetResult(double[][][] input)
        {
            double max = Double.MinValue;
            int index = 0;
            int length = input[0][0].Length;
            for (int i = 0; i < length; i++)
            {
                if (input[0][0][i] > max)
                {
                    max = input[0][0][i];
                    index = i;
                }
            }
            return index;
        }
        
        public override void TestDimension(double[][][][] input, double[][] targets, int batchStart = 0, int batchEnd = 0)
        {
            int numSamples = input.Length;
            int numClasses = input[0][0][0].Length;
            int exNumSamples;
            if (batchEnd == 0)
                exNumSamples = input.Length;
            else
                exNumSamples = batchEnd - batchStart;
            int exNumClasses = targets[0].Length;
            if (numSamples != exNumSamples
                || numClasses != exNumClasses)
            {
                string msg = "Invalid input data in categorical cross entropy, expected input ("
                    + exNumSamples + "," + exNumClasses + ") and found (" + numSamples + "," + numClasses + ")";
                Utils.ThrowException(msg);
            }
        }
    }
}