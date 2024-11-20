// IndexedDB
let indexedDB = window.indexedDB || window.webkitIndexedDB || window.mozIndexedDB || window.OIndexedDB || window.msIndexedDB;
let IDBTransaction = window.IDBTransaction || window.webkitIDBTransaction || window.OIDBTransaction || window.msIDBTransaction;
let dbVersion = 1.0;
let dbName = "FRS";
let storeName = "Dnn";



function createDbConnection(dbName, dbVersion) {
    return new Promise(function (resolve) {
        const request = indexedDB.open(dbName, dbVersion);
        // create the Contacts object store and indexes
        request.onupgradeneeded = (event) => {
            let db = event.target.result;

            // create the object store 
            // with auto-increment id
            let store = db.createObjectStore('Dnn', {
                autoIncrement: true
            });

            // create an index on the email property
            let index = store.createIndex('model_name', 'model_name', {
                unique: true
            });
        };

        request.onsuccess = (event) => {
            var idb = event.target.result;
            resolve(idb);
        };

        request.onerror = (e) => {
            console.log("Enable to access IndexedDB, " + e.target.errorCode)
        };
    });
}

function createTfDbConnection(dbName, dbVersion) {
    return new Promise(function (resolve, reject) {
        const request = indexedDB.open(dbName, dbVersion);
        request.onsuccess = (event) => {
            var idb = event.target.result;
            resolve(idb);
        };

        request.onerror = (e) => {
            showMessage("Unable to access IndexedDB: " + e.target.errorCode);
            reject(undefined);
        };
    });
}

async function insertModel(dbName, dbVersion, storeName, model) {
    let db = await createDbConnection(dbName, dbVersion);

    return new Promise(function (resolve) {
        // create a new transaction
        const txn = db.transaction(storeName, 'readwrite');

        // get the Contacts object store
        const store = txn.objectStore(storeName);
        // put data to store
        let query = store.put(model);

        // handle success case
        query.onsuccess = function (event) {
            console.log(event);
            resolve(event.result);
        };

        // handle the error case
        query.onerror = function (event) {
            console.log(event.target.errorCode);
        }

        // close the database once the 
        // transaction completes
        txn.oncomplete = function () {
            db.close();
        };
    });
}


async function getModelByName(dbName, dbVersion, storeName, indexName, name) {
    // create db connection
    let db = await createDbConnection(dbName, dbVersion);

    return new Promise(function (resolve) {
        // create transaction
        const txn = db.transaction(storeName, 'readonly');

        // create object store by store name
        const store = txn.objectStore(storeName);

        // get the index from the Object Store
        const index = store.index(indexName);
        // query by indexes
        let query = index.get(name);

        // return the result object on success
        query.onsuccess = (event) => {
            // console.log(query.result); // result objects
            resolve(query.result);
        };

        query.onerror = (event) => {
            console.log(event.target.errorCode);
        }

        // close the database connection
        txn.oncomplete = function () {
            db.close();
        };
    });
}

async function isDbExists(dbName) {
    return new Promise(function (resolve) {
        var dbExists = true;
        var request = indexedDB.open(dbName);
        request.onupgradeneeded = function (event) {
            event.target.transaction.abort();
            dbExists = false;
            resolve(false);
        };

        request.onsuccess = (event) => {
            resolve(dbExists);
        };
    });
}


async function getTfModelByName(dbName, dbVersion, storeName, name) {
    let exists = await isDbExists(dbName);
    console.log("Db exists: ", exists);
    if(!exists) return undefined;

    let db = await createTfDbConnection(dbName, dbVersion);

    return new Promise(function (resolve) {
        let stores = db.objectStoreNames;
        console.log(stores);

        if (stores.length === 0 || !stores.contains(storeName)) {
            console.log("no store...");
            resolve(undefined);
        }

        // create transaction
        const txn = db.transaction(storeName, 'readonly');
        console.log(txn);

        // create object store by store name
        const store = txn.objectStore(storeName);

        // get the index from the Object Store
        let query = store.get(name);

        // return the result object on success
        query.onsuccess = (event) => {
            // console.log(query.result); // result objects
            resolve(query.result);
        };

        query.onerror = (event) => {
            console.log(event.target.errorCode);
        }

        // close the database connection
        txn.oncomplete = function () {
            db.close();
        };
    });
}