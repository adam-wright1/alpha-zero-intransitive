const SYMBOLS = {
    blue_rock: '🔵', blue_paper: '🟦', blue_scissors: '🔷',
    red_rock: '🟠', red_paper: '🟧', red_scissors: '🔶',
};

let selected = null;

function renderBoard(data) {
    const boardDiv = document.getElementById('board');
    boardDiv.innerHTML = '';
    const n = data.grid.length;

    for (let row = 0; row < n; row++) {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'row';
        for (let col = 0; col < n; col++) {
            const cell = document.createElement('div');
            cell.className = 'cell';

            if (row === 0 && col === n - 1) cell.classList.add('blue-goal');
            if (row === n - 1 && col === 0) cell.classList.add('red-goal');

            const piece = data.grid[row][col];
            cell.textContent = piece ? SYMBOLS[piece] : '';
            cell.dataset.row = row;
            cell.dataset.col = col;
            cell.addEventListener('click', onCellClick);
            rowDiv.appendChild(cell);
        }
        boardDiv.appendChild(rowDiv);
    }

    document.getElementById('status').textContent = data.game_over
        ? `Game over! Result: ${data.result}`
        : `${data.player === 1 ? 'Your' : "Bot's"} move`;
}

function onCellClick(e) {
    const row = parseInt(e.target.dataset.row);
    const col = parseInt(e.target.dataset.col);

    if (selected === null) {
        selected = [row, col];
        e.target.classList.add('selected');
        return;
    }

    fetch('/move', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({from: selected, to: [row, col]})
    })
        .then(res => res.json())
        .then(data => {
            selected = null;
            document.querySelectorAll('.cell.selected').forEach(c => c.classList.remove('selected'));

            if (data.error) {
                return; // silently cancel, no popup
            }

            renderBoard(data);

            if (!data.game_over) {
                document.getElementById('status').textContent = "Bot is thinking...";
                fetch('/bot_move', {method: 'POST'})
                    .then(res => res.json())
                    .then(renderBoard);
            }
        });
}

document.getElementById('reset-btn').addEventListener('click', () => {
    fetch('/reset', {method: 'POST'}).then(() => fetch('/state')).then(r => r.json()).then(renderBoard);
});

fetch('/state').then(r => r.json()).then(renderBoard);

let bvbActive = false;

document.getElementById('bvb-mode-btn').addEventListener('click', () => {
    document.getElementById('bvb-setup').style.display = 'block';
    fetch('/checkpoints').then(r => r.json()).then(ckpts => {
        const opts = ckpts.map(c => `<option value="${c}">${c}</option>`).join('');
        document.getElementById('checkpoint1').innerHTML = opts;
        document.getElementById('checkpoint2').innerHTML = opts;
    });
});

document.getElementById('next-btn').addEventListener('click', () => {
    fetch('/bvb_forward', {method: 'POST'}).then(r => r.json()).then(renderBvb);
});

document.getElementById('prev-btn').addEventListener('click', () => {
    fetch('/bvb_backward', {method: 'POST'}).then(r => r.json()).then(renderBvb);
});

document.getElementById('start-bvb-btn').addEventListener('click', () => {
    const ckpt1 = document.getElementById('checkpoint1').value;
    const ckpt2 = document.getElementById('checkpoint2').value;

    fetch('/start_bvb', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({checkpoint1: ckpt1, checkpoint2: ckpt2})
    }).then(r => r.json()).then(data => {
        document.getElementById('bvb-setup').style.display = 'none';
        document.getElementById('bvb-nav').style.display = 'block';
        const slider = document.getElementById('move-slider');
        slider.max = data.total_moves;
        slider.value = 1;
        fetch('/bvb_state').then(r => r.json()).then(renderBvb);
    });
});

document.getElementById('move-slider').addEventListener('input', (e) => {
    fetch('/bvb_seek', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({index: parseInt(e.target.value)})
    }).then(r => r.json()).then(renderBvb);
});

function renderBvb(data) {
    renderBoard(data);
    document.getElementById('move-counter').textContent = `Move ${data.move_number} / ${data.total_moves}`;
}