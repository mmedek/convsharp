using Microsoft.VisualStudio.TestTools.UnitTesting;
using Zcu.Convsharp.Common;

namespace convsharpTests
{
    [TestClass]
    public class UtilsTest
    {
        [TestMethod]
        public void Init4dArrTest()
        {
            double[][][][] target = Utils.Init4dArr(5, 4, 3, 2);
            Assert.AreEqual(target.Length, 5);
            Assert.AreEqual(target[0].Length, 4);
            Assert.AreEqual(target[0][0].Length, 3);
            Assert.AreEqual(target[0][0][0].Length, 2);
        }

        [TestMethod]
        public void Init3dArrTest()
        {
            double[][][] target = Utils.Init3dArr(5, 4, 3);
            Assert.AreEqual(target.Length, 5);
            Assert.AreEqual(target[0].Length, 4);
            Assert.AreEqual(target[0][0].Length, 3);
        }

        [TestMethod]
        public void Init2dArrTest()
        {
            double[][] target = Utils.Init2dArr(5, 4);
            Assert.AreEqual(target.Length, 5);
            Assert.AreEqual(target[0].Length, 4);
        }

        [TestMethod]
        public void Init5dIntArrTest()
        {
            int[][][][][] target = Utils.InitInt5dArr(5, 4, 3, 2, 1);
            Assert.AreEqual(target.Length, 5);
            Assert.AreEqual(target[0].Length, 4);
            Assert.AreEqual(target[0][0].Length, 3);
            Assert.AreEqual(target[0][0][0].Length, 2);
            Assert.AreEqual(target[0][0][0][0].Length, 1);
        }
    }
}