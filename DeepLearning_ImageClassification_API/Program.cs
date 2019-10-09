using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Transforms;

namespace DeepLearning_ImageClassification_API
{
    class Program
    {
        static void Main(string[] args)
        {
            var assetsRelativePath = "../../../assets";

            MLContext mlContext = new MLContext();

            IEnumerable<ModelInput> images = LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);

            IDataView imageData = mlContext.Data.LoadFromEnumerable(images);

            IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);

            TrainTestData dataSplit = mlContext.Data.TrainTestSplit(shuffledData);
            IDataView trainSet = dataSplit.TrainSet;
            IDataView validationSet = dataSplit.TestSet;

            var mapLabelEstimator = mlContext.Transforms.Conversion.MapValueToKey
                (outputColumnName: "LabelAsKey",
                    inputColumnName: "Label",
                    keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue);

            IDataView transformedValidationData = mapLabelEstimator
                                        .Fit(validationSet)
                                        .Transform(validationSet);

            var trainingPipeline =
                mapLabelEstimator
                .Append(mlContext.Model.ImageClassification(
                    featuresColumnName: "Image",
                    labelColumnName: "LabelAsKey",
                    arch: ImageClassificationEstimator.Architecture.ResnetV2101,
                    epoch: 100,
                    batchSize: 20,
                    testOnTrainSet: false,
                    metricsCallback: (metrics) => Console.WriteLine(metrics),
                    validationSet: transformedValidationData,
                    reuseTrainSetBottleneckCachedValues: true,
                    reuseValidationSetBottleneckCachedValues: true))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            ITransformer trainedModel = trainingPipeline.Fit(trainSet);

            ClassifyImages(mlContext, validationSet, trainedModel);

            Console.ReadKey();
        }

        public static void ClassifyImages(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            IDataView predictionData = trainedModel.Transform(data);

            IEnumerable<ModelOutput> predictions = mlContext.Data.CreateEnumerable<ModelOutput>(predictionData, reuseRowObject: true).Take(10);

            foreach (var prediction in predictions)
            {
                Console.WriteLine($"Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
            }
        }

        public static IEnumerable<ModelInput> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;

                var label = Path.GetFileName(file);
                if (useFolderNameAsLabel)
                    label = Directory.GetParent(file).Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }

                yield return new ModelInput()
                {
                    ImagePath = file,
                    Image = File.ReadAllBytes(file),
                    Label = label
                };
            }
        }

        private static ReadOnlyMemory<char>[] GetKeyMappings(IDataView predictions)
        {
            VBuffer<ReadOnlyMemory<char>> keys = default;
            predictions.Schema["LabelAsKey"].GetKeyValues(ref keys);
            var originalValues = keys.DenseValues().ToArray();
            return originalValues;
        }
    }
    
    class ModelInput
    {
        public string ImagePath { get; set; }
        public byte[] Image { get; set; }
        public string Label { get; set; }
    }

    class ModelOutput
    {
        public string Label { get; set; }

        public string PredictedLabel { get; set; }
    }
}
