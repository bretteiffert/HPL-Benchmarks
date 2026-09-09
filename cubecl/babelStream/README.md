cargo run --release --features cuda -- --dot-blocks 528
cargo run --release --features hip -- --dot-blocks 480

where dot-blocks is 4 x sm_count

h100, 132 = 528
mi100,120 = 480

