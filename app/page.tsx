"use client";

import { useEffect, useRef, useState } from "react";

type User = {
  id: number;
  name: string;
  age: number;
  watched: string[];
};

type Movie = {
  id: number;
  title: string;
  year: number;
  genre: string;
  duration: number;
  rating: number;
};

export default function Home() {

  const [users, setUsers] = useState<User[]>([])
  const [movies, setMovies] = useState<Movie[]>([]);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    const worker = new Worker("/workers/modelWorker.js", {
      type: "module"
    });

    workerRef.current = worker;

    worker.onmessage = (event) => {
      console.log("Mensagem do Worker:", event.data);

      if (event.data.type === "DATA_LOADED") {
        setUsers(event.data.users)
        setMovies(event.data.movies)
      };
    };

    return () => {
      worker.terminate();
    };
  }, []);


  return (
    <main className="min-h-screen bg-black text-white px-8 py-10">

      <h1 className="text-center font-bold text-4xl mb-8">
        Recomendador de Filmes
      </h1>

      <div className="flex justify-center mb-10">
        <select
          defaultValue=""
          onChange={(event) => {
            const userId = Number(event.target.value);

            workerRef.current?.postMessage({
              type: "SELECT_USER",
              userId: userId,
            });
          }}
          className="w-72 bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-3"
        >
          <option value="" disabled>
            Escolha um usuário
          </option>

          {users.map((user) => (
            <option key={user.id} value={user.id}>
              {user.name}
            </option>
          ))}
        </select>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 max-w-7xl mx-auto">
        {movies.map((movie) => (
          <div
            key={movie.id}
            className="border border-zinc-800 bg-zinc-950 rounded-xl p-5"
          >
            <div className="flex flex-row gap-3">
              <h2 className="font-bold text-xl mb-3">
                {movie.title}
              </h2>
              <p className="text-center text-red-600 font-bold">{movie.id}</p>
            </div>

            <p className="text-zinc-400">
              {movie.genre}
            </p>

            <p className="text-zinc-400">
              {movie.year}
            </p>

            <p className="mt-3 font-semibold">
              ⭐ {movie.rating}
            </p>
          </div>
        ))}
      </div>

    </main>
  )
}