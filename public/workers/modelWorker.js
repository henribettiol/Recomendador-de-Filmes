import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/+esm";

let movies = []
let users = []
let genres = []
let genresIndex = {}
let model = null

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

    const trainData = createTrainingData()

    model = await trainModel(trainData)

    postMessage({
        type: "TRAINING_COMPLETE"
    })

}

onmessage = (event) => {
    if (event.data.type === "SELECT_USER") {
        const userId = event.data.userId

        const user = users.find((user) => user.id === userId)

        if (!model) {
            console.log("Modelo ainda está treinando")
            return
        }

        recommend(user)
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

function createTrainingData() {
    const inputs = []
    const labels = []

    const dimensions = 3 + genres.length

    users.forEach((user) => {
        const userVector = encodeUser(user).dataSync()

        movies.forEach((movie) => {
            const movieVector = encodeMovie(movie).dataSync()

            const label = user.watched.includes(movie.title) ? 1 : 0

            inputs.push([
                ...userVector,
                ...movieVector
            ]) // inputs: 600 vetores, cada um com 28 items dentro, sendo o vetor do usuário, com 14 items, já filtrado e com média: userVector
            //  e o movieVector sendo os outros 14, já normalizados com base nos movies.

            labels.push(label) // labels, são as respostas certas de acordo com o usuário, então vai ser uma matriz com 600 linhas e 1 coluna,
            // e 1 se ele ja viu o filme e 0 se não
        })
    })

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimension: dimensions * 2
    }

}

function createModel(inputDimension) {
    const model = tf.sequential()

    model.add(tf.layers.dense({
        inputShape: [inputDimension],
        units: 128,
        activation: "relu"
    }))

    model.add(tf.layers.dense({
        units: 64,
        activation: "relu"
    }))

    model.add(tf.layers.dense({
        units: 32,
        activation: "relu"
    }))

    model.add(tf.layers.dense({
        units: 1,
        activation: "sigmoid"
    }))

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: "binaryCrossentropy",
        metrics: ["accuracy"]
    })

    return model
}

async function trainModel(trainData) {
    const model = createModel(trainData.inputDimension)

    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(
                    `Epoch ${epoch + 1} | loss: ${logs.loss} | acc: ${logs.acc}`
                )
            }
        }
    })
    return model
}

function recommend(user) {
    if (!model) return

    const userVector = encodeUser(user).dataSync()

    const movieVectors = movies.map((movie) => ({
        movie,
        vector: encodeMovie(movie).dataSync()
    }))

    const input = movieVectors.map(({ vector }) => [
        ...userVector,
        ...vector
    ])

    const inputTensor = tf.tensor2d(input)

    const predictions = model.predict(inputTensor)

    const score = predictions.dataSync()

    const recommendations = movieVectors.map((item, index) => ({
        ...item.movie,
        score: score[index]
    })).filter((movie) => !user.watched.includes(movie.title)).sort((a, b) => b.score - a.score)

    postMessage({
        type: "RECOMMENDATIONS",
        user,
        recommendations
    })
}

loadData()



// primeiros treinamos nosso modelo, utilizamos o loadData(), encodeUser(), encodeMovie(), createTrainingData(), createModel()
// isso tudo pra criar nosso modelo, nossa inteligência com base na nossa base de dados, dando para a rede neural no final, um total de:
// 600 linhas e 28 colunas, sendo 600 divido por 6 usuários, cada usuário/100 linhas com os primeiros 14 vetores iguais, e os outros 14
// sendo o vetor de cada filme.

// depois fizemos a função de recomendação, onde foi nescessário também utilizar o encodeUser(), mas agora iriamos fazer uma predição do vetor
// do nosso usuário escolhido, com base no nosso modelo já treinado. nosso input para predição é: 100 linhas, de 28 colunas, sendo as primeiras
// 14 colunas todas iguais, e só mudando as 14 últimos para cada filme existente.

// então, minha análise foi que de que dividimos em duas partes, criar e treinar o modelo com os dados corretos, e posteriormente fazer uma
// predição com base no modelo criado com a nossa base de dados, e um usuário da nossa base de dados.