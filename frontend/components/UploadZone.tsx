"use client";

import { useCallback, useRef, useState } from "react";

interface Props {
  onFile: (file: File) => void;
  fileName: string | null;
  imageUrl: string | null;
}

export default function UploadZone({ onFile, fileName, imageUrl }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const accept = useCallback(
    (file: File | undefined) => {
      setError(null);
      if (!file) return;
      if (!/image\/(png|jpe?g)/.test(file.type)) {
        setError("Please choose a PNG or JPG image.");
        return;
      }
      onFile(file);
    },
    [onFile],
  );

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          accept(e.dataTransfer.files[0]);
        }}
        onClick={() => inputRef.current?.click()}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-12 text-center transition
          ${dragOver ? "border-indigo-500 bg-indigo-50" : "border-gray-300 bg-white hover:border-indigo-400 hover:bg-gray-50"}`}
      >
        {imageUrl ? (
          <div className="flex flex-col items-center gap-3">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={imageUrl}
              alt="Uploaded artwork"
              className="max-h-48 rounded-lg border border-gray-200 bg-[repeating-conic-gradient(#f3f4f6_0_25%,white_0_50%)] bg-[length:16px_16px] object-contain p-1"
            />
            <p className="text-sm text-gray-600">{fileName}</p>
            <p className="text-xs text-gray-400">
              Click or drop another image to replace it
            </p>
          </div>
        ) : (
          <>
            <div className="mb-3 text-5xl">🪡</div>
            <p className="text-lg font-medium text-gray-700">
              Drop your PNG here
            </p>
            <p className="mt-1 text-sm text-gray-500">
              or click to browse — PNG with transparent background works best
              (JPG is fine too)
            </p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg"
          className="hidden"
          onChange={(e) => accept(e.target.files?.[0])}
        />
      </div>
      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
    </div>
  );
}
