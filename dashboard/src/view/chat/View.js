import { useState } from "react";
import ViewAPI from './ViewAPI'
export default function View({  }) {

    const [inputText, setInputText] = useState("");
    const { messages, onSend } = ViewAPI();

    const handleSend = () => {
        onSend(inputText);
        setInputText("");
    };

    return (
        <div style={{ padding: '20px' }}> 
            <div style={{ height: '300px', overflowY: 'auto', border: '1px solid #ddd', marginBottom: '10px' }}>
                {messages.length === 0 ? (
                    <p>대화를 시작하세요.</p>
                ) : (
                    messages.map(msg => (
                        <div key={msg.id} style={{ textAlign: msg.sender === 'user' ? 'right' : 'left' }}>
                            <p><b>{msg.sender}:</b> {msg.text}</p>
                        </div>
                    ))
                )}
            </div>

            <input
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="메시지 입력"
            />
            <button onClick={handleSend}>전송</button>
        </div>
    );
}