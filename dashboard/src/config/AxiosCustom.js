import axios from "axios";

const baseURL = "http://localhost:6060";

const AxiosCustom = axios.create({
    baseURL: baseURL,
    headers: { "Content-Type": "application/json" },
});


export default AxiosCustom;