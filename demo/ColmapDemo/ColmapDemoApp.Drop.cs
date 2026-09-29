// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ColmapDemoApp.Drop: files dragged onto the window. agg's FileDropDispatcher delivers a hover as a
// mouse move and the drop as a mouse up, both carrying the paths, on every host that has a drop (the
// mac and the browser). On the mac those are the user's own files and are used where they are. The
// browser stages each drop into the wasm file system (agg-file-dialogs/drop-*) and hands ownership
// to the receiver, which must release it (ReleaseDroppedFile) or the tab's memory fills up with every
// drop: so a browser drop's photos are moved into a folder the demo owns (deleted by Clear and when
// the window closes, as video frames are) and the drop is released once its videos are done with.
// The rest of the panel is in ColmapDemoApp.cs; reading videos in ColmapDemoApp.Video.cs.
// Tests: demo/ColmapDemo.Tests/DropIntakeTests.cs.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using MatterHackers.Agg.UI;

namespace ColmapDemo
{
	public partial class ColmapDemoApp
	{
		// The folders a browser drop's photos were moved into, for Clear and close to delete.
		private readonly List<string> droppedPhotoFolders = new List<string>();

		/// <summary>
		/// Set where a drop's files are staged copies that the receiver owns and must free once it is
		/// done with them (the browser head passes <c>BrowserFileStaging.Release</c>); null where a drop
		/// carries the user's own files (the mac). When set, <see cref="ReceiveDropAsync"/> moves the
		/// drop's photos into a folder of the demo's and releases every dropped path after its videos
		/// are read.
		/// </summary>
		public Action<string> ReleaseDroppedFile { get; set; }

		/// <summary>
		/// A drag over the window is taken when it carries at least one photo or video. In the browser
		/// the hover's paths are stand-ins named from the MIME type (dragged-1.mp4), which is enough here.
		/// </summary>
		public override void OnMouseMove(MouseEventArgs mouseEvent)
		{
			if (mouseEvent.DragFiles?.Any(p => IsPhotoPath(p) || IsVideoPath(p)) == true)
			{
				mouseEvent.AcceptDrop = true;
			}

			base.OnMouseMove(mouseEvent);
		}

		/// <summary>The drop itself arrives as a mouse up carrying the files (agg's FileDropDispatcher).</summary>
		public override void OnMouseUp(MouseEventArgs mouseEvent)
		{
			if (mouseEvent.DragFiles?.Count > 0)
			{
				// Not awaited: it reports every outcome in the panel (a failed move included) and never throws.
				_ = this.ReceiveDropAsync(mouseEvent.DragFiles.ToList());
			}

			base.OnMouseUp(mouseEvent);
		}

		/// <summary>
		/// Adds a drop's photos and videos, as <see cref="AddPhotos"/> does. Where a drop's files are
		/// staged (<see cref="ReleaseDroppedFile"/> set), its photos are first moved out into a folder
		/// of the demo's - skipping ones already listed and ones whose staged file is missing, and saying
		/// so in the panel if the move fails - and the whole drop is released once its videos are read
		/// (or were refused, as they are mid-run, or where there is no reader); completes then.
		/// </summary>
		public async Task ReceiveDropAsync(IReadOnlyList<string> paths)
		{
			Action<string> release = this.ReleaseDroppedFile;
			if (release == null)
			{
				await this.AddFilesAsync(paths);
				return;
			}

			try
			{
				// Mid-run nothing is added (AddFilesAsync would ignore it too), so there is nothing to move.
				if (this.IsRunning)
				{
					return;
				}

				(List<string> taken, int missing, string failure) = this.TakeDroppedPhotos(paths);

				// The notes after the add starts: it rewrites the note line, and a video read clears the error line.
				Task videos = this.AddFilesAsync(taken);
				if (missing > 0)
				{
					this.ShowNote(missing == 1 ? "1 dropped photo could not be read and was skipped." : $"{missing} dropped photos could not be read and were skipped.");
				}

				if (failure != null)
				{
					this.ShowError($"Couldn't take the dropped photos: {failure}. Try dropping them again.");
				}

				await videos;
			}
			finally
			{
				foreach (string path in paths)
				{
					release(path);
				}
			}
		}

		/// <summary>
		/// Whether <paramref name="path"/> is a staged copy (a browser drop's or open dialog's) of a photo
		/// already listed. A staged copy has a path of its own every time, so the only sign that it is the
		/// same photo is its name and size; listing it twice would give the mapper a pair with no baseline.
		/// Desktop paths are the user's own files, where the path alone says it.
		/// </summary>
		private bool IsStagedDuplicate(string path)
		{
			if (this.ReleaseDroppedFile == null || !File.Exists(path))
			{
				return false;
			}

			string name = Path.GetFileName(path);
			long size = new FileInfo(path).Length;
			return this.photoPaths.Any(listed => string.Equals(Path.GetFileName(listed), name, StringComparison.Ordinal)
				&& File.Exists(listed) && new FileInfo(listed).Length == size);
		}

		/// <summary>
		/// <paramref name="paths"/> with each photo moved into a new folder under
		/// <see cref="VideoFramesRoot"/>; videos and anything else are left where they are. A move rather
		/// than a copy: in the browser's in-memory file system a copy would hold every photo twice.
		/// Photos already listed are left behind for the release to free, and ones whose staged file is
		/// missing are counted rather than listed as dead paths. A move that fails (a full file system)
		/// stops the taking: the photos moved so far are kept, and the failure is returned for the panel.
		/// </summary>
		private (List<string> Taken, int Missing, string Failure) TakeDroppedPhotos(IReadOnlyList<string> paths)
		{
			var taken = new List<string>(paths.Count);
			int missing = 0;
			string folder = null;
			foreach (string path in paths)
			{
				if (!IsPhotoPath(path))
				{
					taken.Add(path);
					continue;
				}

				if (!File.Exists(path))
				{
					missing++;
					continue;
				}

				if (this.IsStagedDuplicate(path))
				{
					continue;
				}

				try
				{
					if (folder == null)
					{
						folder = Path.Combine(this.VideoFramesRoot, "drop-" + Guid.NewGuid().ToString("N")[..8]);
						Directory.CreateDirectory(folder);
						this.droppedPhotoFolders.Add(folder);
					}

					// One drop can carry two files of the same name from different folders.
					string target = Path.Combine(folder, Path.GetFileName(path));
					for (int suffix = 2; File.Exists(target); suffix++)
					{
						target = Path.Combine(folder, $"{Path.GetFileNameWithoutExtension(path)} ({suffix}){Path.GetExtension(path)}");
					}

					File.Move(path, target);
					taken.Add(target);
				}
				catch (Exception e) when (e is IOException || e is UnauthorizedAccessException)
				{
					Console.WriteLine($"COLMAP_DEMO could not take dropped {path}: {e}");

					// Only the photos stop here; the drop's videos and other files still go on to the add.
					taken.AddRange(paths.Where(p => !IsPhotoPath(p) && !taken.Contains(p)));
					return (taken, missing, FirstLine(e.Message).TrimEnd('.'));
				}
			}

			return (taken, missing, null);
		}

		/// <summary>Deletes the folders a browser drop's photos were moved into.</summary>
		private void DeleteDroppedPhotos()
		{
			foreach (string folder in this.droppedPhotoFolders)
			{
				DeleteFolder(folder);
			}

			this.droppedPhotoFolders.Clear();
		}
	}
}
