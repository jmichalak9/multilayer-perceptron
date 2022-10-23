using MathNet.Numerics.LinearAlgebra;
using OxyPlot;
using OxyPlot.Series;
namespace MultilayerPerceptron;

public class Visualizer
{
    public static void VisualizeClassification(string name, Matrix<double> data, Vector<double> labels)
    {
        var numClasses = (int)labels.Maximum();

        var lineseries = new LineSeries[numClasses];
        var colors = new OxyPlot.OxyColor[] { OxyPlot.OxyColors.Blue, OxyPlot.OxyColors.Red, OxyPlot.OxyColors.Green};
        for (int i = 0; i < numClasses; i++)
        {
            var line = new OxyPlot.Series.LineSeries()
            {
                Title = $"Series {i}",
                Color = colors[i],//OxyPlot.OxyColors.Blue,
                StrokeThickness = 0,
                MarkerSize = 2,
                MarkerType = OxyPlot.MarkerType.Circle
            };
            lineseries[i] = line;
        }

        for (int i = 0; i < data.RowCount; i++)
        {
            var classNum = (int)labels[i] - 1;
            lineseries[classNum].Points.Add(new DataPoint(data[i,0], data[i,1]));
        }

        var model = new OxyPlot.PlotModel
        {
            Title = $"Classification data ({name})"
        };
        foreach (var ls in lineseries)
        {
            model.Series.Add(ls);
        }
        using (var stream = File.Create($"../../../../../{name}.pdf"))
        {
            var pdfExporter = new PdfExporter{Width = 600, Height = 400};
            pdfExporter.Export(model, stream);
        }
    }
}