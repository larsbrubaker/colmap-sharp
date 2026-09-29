// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ModelViewport: the demo's 3D view - the sparse points once the sparse stage is done, then the
// mesh once meshing (and texturing) is done, turned with agg-sharp's TrackballTumbleWidget and
// drawn through RenderGl's SceneDrawContext (the pattern of MatterCAD's LogoSpinner). What it
// draws comes from MeshBridge.cs; ColmapDemoApp.cs owns it and feeds it on the UI thread.
//
// COLMAP's world frame is its first camera's: x right, y down, z into the scene. Flipping y and
// z (a half turn about x) puts the model upright and facing the default agg camera, and the
// model is centered and scaled to a fixed size so the trackball's default distance fits it.

using System;
using System.Collections.Generic;
using System.Linq;
using MatterHackers.Agg;
using MatterHackers.Agg.UI;
using MatterHackers.PolygonMesh;
using MatterHackers.RenderGl;
using MatterHackers.VectorMath;
using MatterHackers.VectorMath.TrackBall;

namespace ColmapDemo
{
	/// <summary>Shows sparse points or a mesh, turned with the mouse.</summary>
	public class ModelViewport : GuiWidget
	{
		/// <summary>The size (in world units) the model is scaled to fill.</summary>
		private const double ModelSize = 4;

		private readonly WorldView world;

		private readonly LightingData lighting = new LightingData();

		private TextWidget hint;

		private PosColorVertex[] pointVertices = Array.Empty<PosColorVertex>();

		private Mesh mesh;

		private Matrix4X4 modelTransform = Matrix4X4.Identity;

		// The scale inside modelTransform, so a point's cross can be sized in screen-ish units.
		private double modelScale = 1;

		public ModelViewport()
		{
			this.Name = "Model Viewport";
			this.HAnchor = HAnchor.Stretch;
			this.VAnchor = VAnchor.Stretch;
			this.BackgroundColor = new Color("#3a3d42");

			this.world = new WorldView(this.Width, this.Height);
			var trackball = new TrackballTumbleWidget(this.world, this)
			{
				TransformState = TrackBallTransformType.Rotation,
			};
			this.AddChild(trackball);

			this.hint = NewHint("No model yet");
			this.AddChild(this.hint);
		}

		/// <summary>
		/// Rebuilds the one thing in here sized at build time, the hint's font, at the current
		/// <see cref="GuiWidget.DeviceScale"/>; the model, its fit and the trackball's view are kept.
		/// ColmapDemoApp.RebuildUi calls it when the display scale changes.
		/// </summary>
		public void RebuildForScale()
		{
			string text = this.hint.Text;
			this.hint.Close();
			this.hint = NewHint(text);
			this.AddChild(this.hint);
		}

		private static TextWidget NewHint(string text) =>
			new TextWidget(text, pointSize: 14, textColor: new Color("#b0b0b0"))
			{
				HAnchor = HAnchor.Center,
				VAnchor = VAnchor.Center,
			};

		/// <summary>Whether a mesh (rather than only points, or nothing) is showing.</summary>
		public bool ShowsMesh => this.mesh != null;

		/// <summary>Shows <paramref name="points"/> (the sparse model) in place of anything shown.</summary>
		public void ShowPoints(IReadOnlyList<ColoredPoint> points)
		{
			this.mesh = null;
			this.modelTransform = FitTransform(points.Select(p => p.Position));

			// Each point is a small 3D cross (three line segments): agg draws lines cheaply and a
			// cross reads as a dot from every side.
			double arm = 0.006 * ModelSize / this.modelScale;
			var vertices = new List<PosColorVertex>(points.Count * 6);
			foreach (ColoredPoint point in points)
			{
				for (int axis = 0; axis < 3; axis++)
				{
					var offset = Vector3.Zero;
					offset[axis] = arm;
					vertices.Add(new PosColorVertex(point.Position - offset, point.Color));
					vertices.Add(new PosColorVertex(point.Position + offset, point.Color));
				}
			}

			this.pointVertices = vertices.ToArray();
			this.hint.Text = points.Count == 0 ? "The photos did not connect into a model" : string.Empty;
			this.Invalidate();
		}

		/// <summary>Shows <paramref name="newMesh"/> in place of the points.</summary>
		public void ShowMesh(Mesh newMesh)
		{
			this.mesh = newMesh;
			this.pointVertices = Array.Empty<PosColorVertex>();
			this.modelTransform = FitTransform(newMesh.Vertices.Select(v => new Vector3(v.X, v.Y, v.Z)));
			this.hint.Text = string.Empty;
			this.Invalidate();
		}

		/// <summary>Back to the empty "No model yet" view.</summary>
		public void Clear()
		{
			this.mesh = null;
			this.pointVertices = Array.Empty<PosColorVertex>();
			this.hint.Text = "No model yet";
			this.Invalidate();
		}

		public override void OnDraw(Graphics2D graphics2D)
		{
			base.OnDraw(graphics2D);
			if (this.mesh == null && this.pointVertices.Length == 0)
			{
				return;
			}

			RectangleDouble screenBounds = this.TransformToScreenSpace(this.LocalBounds);
			var context = SceneDrawContext.TryCreate(graphics2D, this.world);
			if (context == null)
			{
				return;
			}

			context.BeginFrame(this.world, screenBounds, this.lighting);
			if (this.mesh != null)
			{
				context.DrawMesh(this.mesh, Color.White, this.modelTransform, RenderTypes.Shaded, forceCullBackFaces: false);
			}
			else
			{
				context.DrawPrimitives(DrawTopology.LineList, this.pointVertices, this.modelTransform, depthTest: true);
			}

			context.EndFrame();
		}

		// Centered at the origin, scaled to ModelSize, y and z flipped (see the file header).
		// Uses the 5th-95th percentile box so a few stray points do not shrink the model to a dot.
		private Matrix4X4 FitTransform(IEnumerable<Vector3> positions)
		{
			List<Vector3> list = positions.ToList();
			if (list.Count == 0)
			{
				this.modelScale = 1;
				return Matrix4X4.Identity;
			}

			var min = new Vector3(Percentile(list, 0, 0.05), Percentile(list, 1, 0.05), Percentile(list, 2, 0.05));
			var max = new Vector3(Percentile(list, 0, 0.95), Percentile(list, 1, 0.95), Percentile(list, 2, 0.95));
			Vector3 center = (min + max) / 2;
			double extent = Math.Max(max.X - min.X, Math.Max(max.Y - min.Y, max.Z - min.Z));
			double scale = extent > 0 ? ModelSize / extent : 1;
			this.modelScale = scale;
			return Matrix4X4.CreateTranslation(-center)
				* Matrix4X4.CreateScale(scale)
				* Matrix4X4.CreateRotationX(Math.PI);
		}

		private static double Percentile(List<Vector3> points, int axis, double fraction)
		{
			double[] values = points.Select(p => p[axis]).OrderBy(v => v).ToArray();
			return values[(int)Math.Round(fraction * (values.Length - 1))];
		}
	}
}
