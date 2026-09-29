// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PhotoPlacement: which of a run's photos the sparse mapper placed (registered), and the words the
// panel uses to say so. ReconstructionSession.cs builds it from the run's models once the mapper is
// done; ColmapDemoApp.Run.cs shows its StatusText under the stage list and marks the photo list's
// lines (ColmapDemoApp.cs) that were not placed.
//
// The mapper can build several separate models ("groups") when the photos do not all overlap.
// Their coordinate frames are unrelated, so the demo shows and meshes one: the one with the most
// placed photos. That is not always model 0: the incremental mapper keeps models in the order it
// built them, and only ReconstructionManager.Write (the sparse/<i> folders) sorts them, by point
// count, so the in-memory index can differ from the folder index.

using System;
using System.Collections.Generic;
using System.Linq;
using ColmapSharp.Scene;

namespace ColmapDemo
{
	/// <summary>Where one photo of a run ended up.</summary>
	public enum PlacementState
	{
		/// <summary>Registered in the model that is shown and meshed.</summary>
		Shown,

		/// <summary>Registered, but in a smaller separate model that is not shown.</summary>
		OtherGroup,

		/// <summary>Not registered in any model.</summary>
		NotPlaced,
	}

	/// <summary>Which of a run's photos were placed, and in which group.</summary>
	public sealed class PhotoPlacement
	{
		/// <summary>What a user can do about photos that were not placed, in the panel's words.</summary>
		public const string UnplacedAdvice = "Photos that could not be placed probably didn't overlap enough or had too little texture.";

		/// <param name="photoNames">The run's photos by the names the library knew them by, in input order.</param>
		/// <param name="states">Each photo's placement, in the same order.</param>
		/// <param name="groupCount">How many separate models the mapper kept.</param>
		public PhotoPlacement(IReadOnlyList<string> photoNames, IReadOnlyList<PlacementState> states, int groupCount)
		{
			if (photoNames.Count != states.Count)
			{
				throw new ArgumentException("There must be one placement per photo.", nameof(states));
			}

			this.PhotoNames = photoNames;
			this.States = states;
			this.GroupCount = groupCount;
		}

		/// <summary>The run's photos by the names the library knew them by, in input order.</summary>
		public IReadOnlyList<string> PhotoNames { get; }

		/// <summary>Each photo's placement, in the order of <see cref="PhotoNames"/>.</summary>
		public IReadOnlyList<PlacementState> States { get; }

		/// <summary>How many separate models (groups) the mapper kept.</summary>
		public int GroupCount { get; }

		/// <summary>How many photos the run had.</summary>
		public int TotalCount => this.States.Count;

		/// <summary>How many photos were placed in any group.</summary>
		public int PlacedCount => this.States.Count(s => s != PlacementState.NotPlaced);

		/// <summary>How many photos are in the shown group.</summary>
		public int ShownCount => this.States.Count(s => s == PlacementState.Shown);

		/// <summary>
		/// The panel's summary: "Placed N of M photos.", then, with several groups, which one is
		/// shown, and the advice when any photo did not make it into the shown group.
		/// </summary>
		public string StatusText
		{
			get
			{
				string text = $"Placed {this.PlacedCount} of {this.TotalCount} photos.";
				if (this.GroupCount > 1)
				{
					text += $" The photos formed {this.GroupCount} separate groups; showing the largest ({this.ShownCount} photos).";
				}

				if (this.ShownCount < this.TotalCount)
				{
					text += " " + UnplacedAdvice;
				}

				return text;
			}
		}

		/// <summary>
		/// The model with the most registered images, the earlier one on a tie (the mapper builds
		/// its best-initialized model first), or -1 when there is none.
		/// </summary>
		public static int LargestModelIndex(ReconstructionManager models)
		{
			int best = -1;
			for (int i = 0; i < models.Size; i++)
			{
				if (best < 0 || models.Get(i).NumRegImages > models.Get(best).NumRegImages)
				{
					best = i;
				}
			}

			return best;
		}

		/// <summary>
		/// Where each of <paramref name="photoNames"/> ended up in <paramref name="models"/>, with
		/// model <paramref name="shownModel"/> (see <see cref="LargestModelIndex"/>) the one shown.
		/// </summary>
		public static PhotoPlacement FromModels(IReadOnlyList<string> photoNames, ReconstructionManager models, int shownModel)
		{
			var placedIn = new Dictionary<string, int>(StringComparer.Ordinal);
			for (int m = 0; m < models.Size; m++)
			{
				Reconstruction model = models.Get(m);
				foreach (uint imageId in model.RegImageIds())
				{
					string name = model.Image(imageId).Name;

					// A photo is only in one model, but the shown one wins if that ever changes.
					if (!placedIn.ContainsKey(name) || m == shownModel)
					{
						placedIn[name] = m;
					}
				}
			}

			var states = photoNames
				.Select(n => !placedIn.TryGetValue(n, out int m) ? PlacementState.NotPlaced : m == shownModel ? PlacementState.Shown : PlacementState.OtherGroup)
				.ToList();
			return new PhotoPlacement(photoNames, states, models.Size);
		}
	}
}
