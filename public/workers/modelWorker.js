import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/+esm";

let movies = []
let users = []
let genres = []
let genresIndex = {}

async function loadData() {
    const movieResponse = await fetch('/data/movies.json')
    const usersResponse = await fetch('/data/users.json')

    movies = await movieResponse.json()
    users = await usersResponse.json()

    genres = [... new Set(movies.map((movie) => movie.genre))]

    genresIndex = Object.fromEntries(
        genres.map((genre, index) => [genre, index])
    )

    postMessage({
        type: "DATA_LOADED",
        moviesCount: movies.length,
        usersCount: users.length,
        users: users,
        movies: movies,
    })

}

onmessage = (event) => {
    if (event.data.type === "SELECT_USER") {
        const userId = event.data.userId

        const user = users.find((user) => user.id === userId)

        const encodedUser = encodeUser(user)

        console.log(encodedUser.dataSync())
    }
};



function normalize(value, min, max) {
    return (value - min) / (max - min)
}

function oneHotWeighted(index, length, weight) {
    return tf.oneHot(index, length)
        .cast("float32")
        .mul(weight)
}

function encodeMovie(movie) {
    const yearsMovies = movies.map((m) => m.year)
    const moviesDurations = movies.map((d) => d.duration)
    const ratingsMovies = movies.map(m => m.rating)

    const minYear = Math.min(...yearsMovies)
    const maxYear = Math.max(...yearsMovies)

    const minDuration = Math.min(...moviesDurations)
    const maxDuration = Math.max(...moviesDurations)

    const minRating = Math.min(...ratingsMovies)
    const maxRating = Math.max(...ratingsMovies)


    const year = tf.tensor1d([
        normalize(movie.year, minYear, maxYear)
    ])


    const duration = tf.tensor1d([
        normalize(movie.duration, minDuration, maxDuration)
    ])

    const rating = tf.tensor1d([
        normalize(movie.rating, minRating, maxRating)
    ])

    const genre = oneHotWeighted(genresIndex[movie.genre], genres.length, 1)

    return tf.concat([
        year,
        duration,
        rating,
        genre
    ])

}

function encodeUser(user) {
    const watchedMovies = movies.filter((movie) =>
        user.watched.includes(movie.title)
    )

    const watchedVectors = watchedMovies.map((movie) =>
        encodeMovie(movie)
    )

    const dimensions = 3 + genres.length

    return tf.stack(watchedVectors).mean(0).reshape([1, dimensions])
}


loadData()